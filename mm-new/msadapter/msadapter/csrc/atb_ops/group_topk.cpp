/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ms_extension/api.h"

using GroupTopkParam = atb::infer::GroupTopkParam;
namespace atb {
template <>
struct HashOpParam<GroupTopkParam> {
  void operator()(const GroupTopkParam &param) const {
    add_param_to_buf("groupNum", param.groupNum);
    add_param_to_buf("k", param.k);
    add_param_to_buf("groupMultiFlag", param.groupMultiFlag);
    add_param_to_buf("n", param.n);
  }
};
}

void npu_group_topk_out(const ms::Tensor &self, int64_t k, int64_t group_num, int64_t n) {
  GroupTopkParam param;
  param.groupNum = static_cast<int32_t>(group_num);
  param.k = static_cast<int32_t>(k);
  param.n = static_cast<uint16_t>(n);

  std::vector<int64_t> v(1024);
  std::iota(v.begin(), v.end(), 0);
  auto idx = ms::tensor(v, ms::TypeId::kNumberTypeInt32);
  ms::pynative::RunAtbOp("GroupTopK", param, {self, idx}, {self});
}

auto pyboost_group_topk(const ms::Tensor &self, int64_t k, int64_t group_num, int64_t n, ms::Tensor out) {
  return ms::pynative::PyboostRunner::Call<0>(npu_group_topk_out, self, k, group_num, n);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_group_topk", &pyboost_group_topk, "GroupTopK", pybind11::arg("self"), pybind11::arg("k"),
        pybind11::arg("group_num"), pybind11::arg("n"), pybind11::arg("out"));
}
