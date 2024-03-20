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

#ifndef XLA_SERVICE_SCAN_LOOP_ACCUMULATOR_INPUT_UNIFICATION_H_
#define XLA_SERVICE_SCAN_LOOP_ACCUMULATOR_INPUT_UNIFICATION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
// This pass looks at the loops with accumulator patterns and unifies the
// accumulation buffer with the input. The accumulation pattern usually comes
// from jax.scan function. This transformation is beneficial in the cases where
// the scan loop appears inside a loop body which causes a copy of the
// accumulation buffer in the outer body. We see an example of jax.scan program
// and its hlo version below: def cumsum(res, elem):
//     """
//     - `res`: The result from the previous loop.
//     - `elem`: The current array element.
//     """
//     res = res + elem
//     return res, res  # ("carryover", "accumulated")
//
// a = np.array([1, 2, 3, 5, 7, 11, 13, 17])
// result_init = 0
// final, result = lax.scan(cumsum, result_init, a)
// result
//
// Below is the body computation of the scan loop:
//  body {
//   param = (s32[], s32[], s32[8], s32[8]) parameter(0)
//   i = s32[] gte(param), index=0
//   init = s32[] gte(param), index=1
//   carry = s32[8] gte(param), index=2
//   input = s32[8] gte(param), index=3
//   dynamic-slice = s32[1] dynamic-slice(input, i), dynamic_slice_sizes={1}
//   new_value = f(dynamic-slice, init) // f is the cumsum function
//   dynamic-update-slice = s32[8] dynamic-update-slice(carry, new_value, i)
//   ROOT tuple.10 = (s32[], s32[], s32[8], s32[8]) tuple(i++, add.1,
//   dynamic-update-slice, input)
//  }
//
// By unifying the accumulation and input buffer we get the following body:
//  body {
//   param = (s32[], s32[], s32[8], s32[8]) parameter(0)
//   i = s32[] gte(param), index=0
//   init = s32[] gte(param), index=1
//   carry = s32[8] gte(param), index=2
//   dynamic-slice = s32[1] dynamic-slice(carry, i), dynamic_slice_sizes={1}
//   new_value = f(dynamic-slice, init) // f is the cumsum function
//   dynamic-update-slice = s32[8] dynamic-update-slice(carry, new_value, i)
//   ROOT tuple.10 = (s32[], s32[], s32[8], s32[8]) tuple(i++, add.1,
//   dynamic-update-slice)
//  }
class ScanLoopAccumulatorInputUnification : public HloModulePass {
 public:
  ~ScanLoopAccumulatorInputUnification() override = default;

  explicit ScanLoopAccumulatorInputUnification() = default;

  absl::string_view name() const override {
    return "scan_loop_accumulator_input_unification";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_SCAN_LOOP_ACCUMULATOR_INPUT_UNIFICATION_H_
