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
struct HashOpParam<atb::infer::ReshapeAndCacheParam> {
  void operator()(const atb::infer::ReshapeAndCacheParam &param) const {
    add_param_to_buf("compressType", param.compressType);
    add_param_to_buf("kvCacheCfg", param.kvCacheCfg);
  }
};
}  // namespace atb

using ReshapeAndCacheParam = atb::infer::ReshapeAndCacheParam;

void reshape_and_cache(const ms::Tensor &key, const std::optional<ms::Tensor> &value, const ms::Tensor &key_cache,
                       const std::optional<ms::Tensor> &value_cache, const ms::Tensor &slot_indices,
                       const int64_t compress_type, const int64_t kvcache_cfg) {
  ReshapeAndCacheParam param;
  param.compressType = static_cast<ReshapeAndCacheParam::CompressType>(compress_type);
  param.kvCacheCfg = static_cast<ReshapeAndCacheParam::KvCacheCfg>(kvcache_cfg);
  std::vector<ms::Tensor> inputs, outputs;
  if (param.kvCacheCfg == ReshapeAndCacheParam::K_CACHE_V_CACHE) {
    ms::Tensor val = value.has_value() ? value.value() : ms::Tensor();
    ms::Tensor vc = value_cache.has_value() ? value_cache.value() : ms::Tensor();
    inputs = std::vector<ms::Tensor>{key, val, key_cache, vc, slot_indices};
    outputs = std::vector<ms::Tensor>{key_cache, vc};
  } else {
    inputs = std::vector<ms::Tensor>{key, key_cache, slot_indices};
    outputs.push_back(key_cache);
  }
  ms::pynative::RunAtbOp("ReshapeAndCache", param, inputs, outputs);
}

auto pyboost_reshape_and_cache(const ms::Tensor &key, const std::optional<ms::Tensor> &value, const ms::Tensor &key_cache,
                       const std::optional<ms::Tensor> &value_cache, const ms::Tensor &slot_indices,
                       const int64_t compress_type, const int64_t kvcache_cfg) {
  return ms::pynative::PyboostRunner::Call<0>(reshape_and_cache, key, value, key_cache, value_cache, slot_indices,
                                              compress_type, kvcache_cfg);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_reshape_cache", &pyboost_reshape_and_cache, "ReshapeAndCache", pybind11::arg("key"), pybind11::arg("value"),
        pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("slot_indices"),
        pybind11::arg("compress_type") = 0, pybind11::arg("kvcache_cfg") = 0);
}
