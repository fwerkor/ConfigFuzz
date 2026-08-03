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

#include <vector>
#include "ms_extension/api.h"

using LinearParam = atb::infer::LinearParam;

std::vector<int64_t> g_shape_hash;

namespace atb {
template <>
struct HashOpParam<LinearParam> {
  void operator()(const LinearParam &param) const {
    add_param_to_buf("transposeA", param.transposeA);
    add_param_to_buf("transposeB", param.transposeB);
    add_param_to_buf("hasBias", param.hasBias);
    add_param_to_buf("enAccum", param.enAccum);
    for (size_t i = 0; i < g_shape_hash.size(); i++) {
      add_param_to_buf("shape_" + std::to_string(i), g_shape_hash[i]);
    }
  }
};
}  // namespace atb

void matmul_add_fp32(const ms::Tensor &x, const ms::Tensor &weight, ms::Tensor C) {
  LinearParam param;
  param.transposeA = true;
  param.transposeB = false;
  param.hasBias = false;
  param.enAccum = true;

  auto x_shape = x.shape();
  if (x_shape.size() == 2 && x_shape[1] > 65535) {
    // In this case, when the shape of the operator executed each time is different,
    // reusing the same Operation will result in an error message "input num is invalid".
    // This is a bug in the atb operator.
    // Here we take a workaround to add the shape to the hash value calculation
    // to ensure that different shapes use different Operations.
    g_shape_hash = weight.shape();
    (void)g_shape_hash.insert(g_shape_hash.end(), x_shape.begin(), x_shape.end());
  } else {
    g_shape_hash.clear();
  }
  ms::pynative::RunAtbOp("MatMulAddFp32", param, {x, weight, C}, {C});
}

auto pyboost_matmul_add_fp32(const ms::Tensor &x, const ms::Tensor &weight, ms::Tensor C) {
  return ms::pynative::PyboostRunner::Call<0>(matmul_add_fp32, x, weight, C);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_matmul_add_fp32", &pyboost_matmul_add_fp32, "matmul_add on ascend device",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("C"));
}
