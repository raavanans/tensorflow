/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/scan_loop_accumulator_input_unification.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/while_loop_unroller.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

bool MatchDynamicSliceInDim(HloInstruction* ds, const WhileLoopConfig& config) {
  // Check that the instruction is a dynamic-slice and populate the input.
  HloInstruction* to_be_sliced;
  if (!Match(ds,
             match::DynamicSlice().WithOperand(0, match::Op(&to_be_sliced)))) {
    return false;
  }

  if (!Match(to_be_sliced, match::GetTupleElement())) {
    return false;
  }

  int64_t ds_dim = -1;
  for (int64_t operand_index = 1; operand_index < ds->operand_count();
       ++operand_index) {
    HloInstruction* operand = ds->mutable_operand(operand_index);
    // All constants must be zero in order to slice the entire shape.
    if (Match(operand, match::ConstantScalar())) {
      std::optional<int64_t> offset =
          LiteralUtil::LiteralAsScalarInt64(operand->literal());
      if (offset.value() != 0) {
        ds_dim = -1;
        break;
      }
    }

    // Check that the slice offset is the loop induction variable.
    if (Match(operand, match::GetTupleElement(match::Parameter(),
                                              config.induction_var_idx))) {
      ds_dim = operand_index - 1;
    }
  }

  if (ds_dim == -1) {
    return false;
  }

  // The shape's broadcast_dim must be exactly equal to the loop trip count.
  if (to_be_sliced->shape().dimensions(ds_dim) != config.trip_count) {
    return false;
  }

  return true;
}

// Check if `instr` is a dus that writes to the entire shape of the `user`. To
// satisfy this, all start indices of the dus must be constant zero except a
// single dimension. The start index of that dimension should be equal to the
// enclosing loop induction variable. Also, the size of that dimension must
// match the loop trip count.
bool MatchDynamicUpdateSliceInDim(HloInstruction* dus, HloInstruction* user,
                                  const WhileLoopConfig& config) {
  // Check that the DUS is a DynamicUpdateSlice.
  HloInstruction* to_be_updated;
  if (!Match(dus, match::DynamicUpdateSlice().WithOperand(
                      0, match::Op(&to_be_updated)))) {
    return false;
  }
  if (to_be_updated != user) {
    return false;
  }

  int64_t dus_dim = -1;
  for (int64_t operand_index = 2; operand_index < dus->operand_count();
       ++operand_index) {
    HloInstruction* operand = dus->mutable_operand(operand_index);
    // All constants must be zero in order to write the entire shape.
    if (Match(operand, match::ConstantScalar())) {
      std::optional<int64_t> offset =
          LiteralUtil::LiteralAsScalarInt64(operand->literal());
      if (offset.value() != 0) {
        dus_dim = -1;
        break;
      }
    }
    // Check that the update offset is the loop induction variable.
    if (Match(operand, match::GetTupleElement(match::Parameter(),
                                              config.induction_var_idx))) {
      dus_dim = operand_index - 2;
    }
  }

  if (dus_dim == -1) {
    return false;
  }

  // The shape's broadcast_dim must be exactly equal to the loop trip count.
  if (user->shape().dimensions(dus_dim) != config.trip_count) {
    return false;
  }

  return true;
}

// This function checks whether the operand of the loop at the given index is
// read-only.
bool LoopIndexIsReadOnly(const HloAliasAnalysis& alias_analysis,
                         HloInstruction* while_instr, int64_t idx) {
  const HloDataflowAnalysis& dataflow_analysis =
      alias_analysis.dataflow_analysis();
  return !(
      dataflow_analysis.GetValueSet(while_instr->while_init(), {idx})
              .values()
              .size() > 1 ||
      dataflow_analysis.GetValueSet(while_instr, {idx}).values().size() > 1 ||
      dataflow_analysis.GetUniqueValueAt(while_instr, {idx}) !=
          dataflow_analysis.GetUniqueValueAt(while_instr->while_init(), {idx}));
}

// This function finds the pairs of accumulator-input pairs in the loop.
// An accumulator-input pair is a pair of instructions that satisfy the
// following conditions:
// 1. The accumulator is updated in the loop body with a dynamic-update-slice
// instruction that covers the whole shape (see the comment for
// MatchDynamicUpdateSliceInDim function).
// 2. The second instruction is read-only in the loop body.
// 3. The two instructions have the same shape.
std::vector<std::pair<HloInstruction*, HloInstruction*>>
FindAccumulatorInputPairs(const HloAliasAnalysis& alias_analysis,
                          HloInstruction* while_instr,
                          const WhileLoopConfig& config) {
  HloComputation* computation = while_instr->while_body();
  HloInstruction* body_param = computation->parameter_instruction(0);

  // Finding the accumulator instructions
  std::vector<HloInstruction*> possible_acc;
  for (int64_t param_idx = 0;
       param_idx < while_instr->while_init()->operand_count(); ++param_idx) {
    for (HloInstruction* gte : body_param->users()) {
      if (!Match(gte, match::GetTupleElement().WithTupleIndex(param_idx))) {
        continue;
      }
      if (gte->operand(0) != body_param) {
        continue;
      }
      // The accumulator should only be updated.
      if (gte->user_count() > 1) {
        continue;
      }
      for (HloInstruction* gte_user : gte->users()) {
        if (MatchDynamicUpdateSliceInDim(gte_user, gte, config)) {
          // The accumulator should be written at the same index
          if (computation->root_instruction()->mutable_operand(param_idx) ==
              gte_user) {
            possible_acc.push_back(gte);
            VLOG(3) << "accumulator index: " << param_idx
                      << ", shape = " << gte->shape().ToString() << gte->name()
                      << ", update_value = "
                      << gte_user->mutable_operand(1)->name();
          }
        }
      }
    }
  }

  // Finding the input indices
  std::vector<HloInstruction*> possible_inputs;
  for (int64_t param_idx = 0;
       param_idx < while_instr->while_init()->operand_count(); ++param_idx) {
    for (HloInstruction* gte : body_param->users()) {
      if (!Match(gte, match::GetTupleElement().WithTupleIndex(param_idx))) {
        continue;
      }
      if (gte->operand(0) != body_param) {
        continue;
      }

      // The input should only be sliced and passed to the next iteration.
      if (gte->user_count() > 2) {
        continue;
      }

      for (HloInstruction* gte_user : gte->users()) {
        if (MatchDynamicSliceInDim(gte_user, config)) {
          // The input should be read-only
          if (LoopIndexIsReadOnly(alias_analysis, while_instr,
                                  gte->tuple_index())) {
            possible_inputs.push_back(gte);
            VLOG(3) << "input" << " index: " << param_idx
                      << ", shape = " << gte->shape().ToString() << gte->name();
          }
        }
      }
    }
  }

  std::vector<std::pair<HloInstruction*, HloInstruction*>> acc_input_pairs;
  std::vector<HloInstruction*> unique_inputs;
  for (HloInstruction* acc : possible_acc) {
    for (HloInstruction* input : possible_inputs) {
      if (ShapeUtil::Equal(input->shape(), acc->shape())) {
        // Make sure all the inputs are unique, if we encounter a used input, we
        // move over to the next candidate.
        if (absl::c_find(unique_inputs, input) != unique_inputs.end()) {
          continue;
        }
        unique_inputs.push_back(input);
        acc_input_pairs.emplace_back(acc, input);
        break;
      }
    }
  }
  return acc_input_pairs;
}

// IsWhileBodyComputation api call does not work properly so we check it
// ourself.
bool IsWhileBody(HloComputation* comp) {
  HloModule* module = comp->parent();
  for (HloComputation* c : module->computations()) {
    for (HloInstruction* instr : c->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile && instr->while_body() == comp) {
        return true;
      }
    }
  }
  return false;
}

// Given a list of unollable loops and their config, remove the unnecessary
// accumulator buffers and replace them with the read-only inputs.
absl::StatusOr<bool> UnifyAccumulatorWithInput(
    const HloAliasAnalysis& alias_analysis,
    std::vector<std::pair<HloInstruction*, WhileLoopConfig>> unrollable_loops) {
  std::vector<HloInstruction*> changed_loops;
  bool removed = false;
  for (auto& [while_instr, loop_config] : unrollable_loops) {
    // We only consider nested loops. The overhead of doing copy where there is
    // not nesting is considered to be negligible.
    if (!IsWhileBody(while_instr->parent())) {
      continue;
    }
    auto acc_input_pairs =
        FindAccumulatorInputPairs(alias_analysis, while_instr, loop_config);
    for (const auto& [acc, input] : acc_input_pairs) {
      // We only consider accumulators that are allocated inside the loop.
      // Therefore, we skip accumulators that are passed as the loop input.
      if (Match(while_instr->while_init()->mutable_operand(acc->tuple_index()),
                match::GetTupleElement(match::Parameter()))) {
        continue;
      }
      VLOG(3) << while_instr->name() << " -> "
              << "<accumulators: " << acc->name() << ", "
              << "input: " << input->name() << ">";
      TF_RETURN_IF_ERROR(input->ReplaceAllUsesWith(acc));
      TF_RETURN_IF_ERROR(while_instr->while_init()->ReplaceOperandWith(
          acc->tuple_index(),
          while_instr->while_init()->mutable_operand(input->tuple_index())));
      if (input->user_count() == 0) {
        TF_RETURN_IF_ERROR(while_instr->while_body()->RemoveInstruction(input));
        removed = true;
      }
    }
  }
  return removed;
}

}  // namespace

absl::StatusOr<bool> ScanLoopAccumulatorInputUnification::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "HLO module before ScanLoopAccumulatorInputUnification:";
  XLA_VLOG_LINES(2, module->ToString());

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module));

  std::vector<std::pair<HloInstruction*, WhileLoopConfig>> unrollable_loops =
      GetUnrollableLoops(module, execution_threads);

  // TODO: we might want to simplify compare instructions before this. It helps
  // us identify more inputs and accumulators.
  TF_ASSIGN_OR_RETURN(bool changed, UnifyAccumulatorWithInput(
                                        *alias_analysis, unrollable_loops));

  if (changed) {
    for (auto& [while_instr, loop_config] : unrollable_loops) {
      TF_RETURN_IF_ERROR(TryRemoveDeadWhileParams(while_instr).status());
    }
    TF_RETURN_IF_ERROR(TupleSimplifier{}.Run(module).status());
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());

    VLOG(2) << "HLO module after ScanLoopAccumulatorInputUnification:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after ScanLoopAccumulatorInputUnification";
  }

  return changed;
}

}  // namespace xla
