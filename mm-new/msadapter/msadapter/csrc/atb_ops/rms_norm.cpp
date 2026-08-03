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

namespace atb {
template <>
struct HashOpParam<atb::infer::RmsNormParam> {
  void operator()(const atb::infer::RmsNormParam &param) const {
    add_param_to_buf("epsilon", param.normParam.epsilon);
    add_param_to_buf("layerType", param.layerType);
    add_param_to_buf("quantType", param.normParam.quantType);
  }
};
}

const static int RMSNORM_LAYERTYPE = 1;

ShapeVector InferShapeRmsNorm(const ShapeVector &self, const ShapeVector &gamma) {
  int64_t rstd_dim = self.size();
  rstd_dim -= gamma.size();
  ShapeVector ret;
  for (size_t i = 0; i < self.size(); i++) {
    if (i < rstd_dim) {
      ret.emplace_back(self[i]);
    } else {
      ret.emplace_back(1);
    }
  }
  return ret;
}

ms::Tensor npu_rms_norm(const ms::Tensor &x, const ms::Tensor &gamma, float epsilon) {
  auto tensor_rstd_shape = InferShapeRmsNorm(x.shape(), gamma.shape());
  ms::Tensor tensor_rstd = ms::Tensor(mindspore::kNumberTypeFloat32, tensor_rstd_shape);
  ms::Tensor tensor_y = ms::Tensor(x.data_type(), x.shape());

  atb::infer::RmsNormParam param;
  param.layerType = (atb::infer::RmsNormParam::RmsNormType)RMSNORM_LAYERTYPE;
  param.normParam.epsilon = epsilon;
  param.normParam.rstd = true;

  ms::pynative::RunAtbOp("RmsNorm", param, {x, gamma}, {tensor_y, tensor_rstd});
  return tensor_y;
}

auto pyboost_npu_rms_norm(const ms::Tensor &x, const ms::Tensor &gamma, float epsilon) {
    return ms::pynative::PyboostRunner::Call<0>(npu_rms_norm, x, gamma, epsilon);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_rms_norm", &pyboost_npu_rms_norm, "RmsNorm", pybind11::arg("x"), pybind11::arg("gamma"), pybind11::arg("epsilon"));
}
