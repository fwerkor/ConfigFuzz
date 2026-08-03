// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ms_extension/api.h"

using GroupedMatmulInplaceAddParam = atb::infer::GroupedMatmulInplaceAddParam;

namespace atb {
template <>
struct HashOpParam<GroupedMatmulInplaceAddParam> {
  void operator()(const GroupedMatmulInplaceAddParam &param) const {
    add_param_to_buf("transposeA", param.transposeA);
    add_param_to_buf("transposeB", param.transposeB);
  }
};
} //namespace atb

void groupmatmul_add_fp32(const ms::Tensor &x, const ms::Tensor &weight, const ms::Tensor &group_list,
                          const ms::Tensor &grad) {
  GroupedMatmulInplaceAddParam param;
  param.transposeA = true;
  param.transposeB = false;
  ms::pynative::RunAtbOp("GroupedMatMulAdd", param, {x, weight, group_list, grad}, {grad});
}

auto pyboost_groupmatmul_add_fp32(const ms::Tensor &x, const ms::Tensor &weight, const ms::Tensor &group_list,
                          const ms::Tensor &grad) {
  return ms::pynative::PyboostRunner::Call<0>(groupmatmul_add_fp32, x, weight, group_list, grad);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_groupmatmul_add_fp32", &pyboost_groupmatmul_add_fp32, "matmul_add on ascend device",
      pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("group_list"), pybind11::arg("grad"));
}
