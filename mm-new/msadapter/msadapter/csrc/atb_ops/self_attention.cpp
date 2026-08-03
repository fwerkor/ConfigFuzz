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
struct HashOpParam<atb::infer::SelfAttentionParam> {
  void operator()(const atb::infer::SelfAttentionParam &param) const {
    add_param_to_buf("calcType", param.calcType);
    add_param_to_buf("kernelType", param.kernelType);
    add_param_to_buf("clampType", param.clampType);
    add_param_to_buf("maskType", param.maskType);
    add_param_to_buf("kvcacheCfg", param.kvcacheCfg);
    add_param_to_buf("scaleType", param.scaleType);
    add_param_to_buf("quantType", param.quantType);
    add_param_to_buf("cacheType", param.cacheType);
    add_param_to_buf("outDataType", param.outDataType);
    add_param_to_buf("headNum", param.headNum);
    add_param_to_buf("kvHeadNum", param.kvHeadNum);
    add_param_to_buf("qScale", param.qScale);
    add_param_to_buf("qkScale", param.qkScale);
    add_param_to_buf("batchRunStatusEnable", param.batchRunStatusEnable);
    add_param_to_buf("isTriuMask", param.isTriuMask);
    add_param_to_buf("clampMin", param.clampMin);
    add_param_to_buf("clampMax", param.clampMax);
    add_param_to_buf("inputLayout", param.inputLayout);
    add_param_to_buf("mlaVHeadSize", param.mlaVHeadSize);
    add_param_to_buf("windowSize", param.windowSize);
  }
};
}  // namespace atb

using SelfAttentionParam = atb::infer::SelfAttentionParam;

void npu_selfattention(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
                       const std::optional<ms::Tensor> &mask, const ms::Tensor &seqLen, const double scale,
                       const int64_t headNum, const int64_t kvHeadNum, ms::Tensor out, int64_t calcType,
                       int64_t kernelType, int64_t clampType, int64_t maskType, int64_t kvcacheCfg, int64_t scaleType,
                       int64_t quantType, int64_t cacheType, int64_t outDataType, double qScale,
                       bool batchRunStatusEnable, int64_t isTriuMask, double clampMin, double clampMax,
                       int64_t inputLayout, int64_t mlaVHeadSize, int64_t windowSize) {
  SelfAttentionParam selfattentionparam;

  auto calctype = static_cast<SelfAttentionParam::CalcType>(calcType);
  selfattentionparam.calcType = calctype;
  auto kerneltype = static_cast<SelfAttentionParam::KernelType>(kernelType);
  selfattentionparam.kernelType = kerneltype;
  auto clamptype = static_cast<SelfAttentionParam::ClampType>(clampType);
  selfattentionparam.clampType = clamptype;
  auto masktype = static_cast<SelfAttentionParam::MaskType>(maskType);
  selfattentionparam.maskType = masktype;
  auto kvcachecfg = static_cast<SelfAttentionParam::KvCacheCfg>(kvcacheCfg);
  selfattentionparam.kvcacheCfg = kvcachecfg;
  auto scaletype = static_cast<SelfAttentionParam::ScaleType>(scaleType);
  selfattentionparam.scaleType = scaletype;
  auto quanttype = static_cast<SelfAttentionParam::QuantType>(quantType);
  selfattentionparam.quantType = quanttype;
  auto cachetype = static_cast<SelfAttentionParam::CacheType>(cacheType);
  selfattentionparam.cacheType = cachetype;
  auto outdataType = static_cast<aclDataType>(outDataType);
  selfattentionparam.outDataType = outdataType;
  selfattentionparam.headNum = headNum;
  selfattentionparam.kvHeadNum = kvHeadNum;
  selfattentionparam.qScale = qScale;
  selfattentionparam.qkScale = scale;
  selfattentionparam.batchRunStatusEnable = batchRunStatusEnable;
  selfattentionparam.isTriuMask = isTriuMask;
  selfattentionparam.clampMin = clampMin;
  selfattentionparam.clampMax = clampMax;
  auto inputlayout = static_cast<atb::infer::InputLayout>(inputLayout);
  selfattentionparam.inputLayout = inputlayout;
  selfattentionparam.mlaVHeadSize = mlaVHeadSize;
  selfattentionparam.windowSize = windowSize;

  std::vector<ms::Tensor> inputs;
  if (!mask.has_value() && selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_UNDEFINED) {
    inputs = std::vector<ms::Tensor>{query, key, value, seqLen};
  } else {
    ms::Tensor mask_t = mask.has_value() ? mask.value() : ms::Tensor();
    inputs = std::vector<ms::Tensor>{query, key, value, mask_t, seqLen};
  }
  ms::pynative::RunAtbOp("Selfattention", selfattentionparam, inputs, {out});
}

auto pyboost_npu_selfattention(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
                       const std::optional<ms::Tensor> &mask, const ms::Tensor &seqLen, const double scale,
                       const int64_t headNum, const int64_t kvHeadNum, ms::Tensor out, int64_t calcType,
                       int64_t kernelType, int64_t clampType, int64_t maskType, int64_t kvcacheCfg, int64_t scaleType,
                       int64_t quantType, int64_t cacheType, int64_t outDataType, double qScale,
                       bool batchRunStatusEnable, int64_t isTriuMask, double clampMin, double clampMax,
                       int64_t inputLayout, int64_t mlaVHeadSize, int64_t windowSize) {
  return ms::pynative::PyboostRunner::Call<0>(npu_selfattention, query, key, value, mask, seqLen, scale, headNum,
      kvHeadNum, out, calcType, kernelType, clampType, maskType, kvcacheCfg, scaleType, quantType, cacheType,
      outDataType, qScale, batchRunStatusEnable, isTriuMask, clampMin, clampMax, inputLayout, mlaVHeadSize, windowSize);
}

void _npu_flash_attention(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
    const ms::Tensor &mask, const ms::Tensor &seq_len, const double scale_value, const int64_t num_heads,
    const int64_t num_kv_heads, ms::Tensor out)
{
    SelfAttentionParam selfattentionparam;

    selfattentionparam.calcType = SelfAttentionParam::PA_ENCODER;
    selfattentionparam.kernelType = SelfAttentionParam::KERNELTYPE_DEFAULT;
    selfattentionparam.clampType = SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
    selfattentionparam.maskType = SelfAttentionParam::MASK_TYPE_NORM;
    selfattentionparam.kvcacheCfg = SelfAttentionParam::K_CACHE_V_CACHE;
    selfattentionparam.scaleType = SelfAttentionParam::SCALE_TYPE_TOR;
    selfattentionparam.quantType = SelfAttentionParam::TYPE_QUANT_UNDEFINED;
    selfattentionparam.cacheType = SelfAttentionParam::CACHE_TYPE_NORM;
    selfattentionparam.outDataType = ACL_DT_UNDEFINED;
    selfattentionparam.headNum = num_heads;
    selfattentionparam.kvHeadNum = num_kv_heads;
    selfattentionparam.qScale = 1;
    selfattentionparam.qkScale = scale_value;
    selfattentionparam.batchRunStatusEnable = false;
    selfattentionparam.isTriuMask = 0;
    selfattentionparam.clampMin = 0;
    selfattentionparam.clampMax = 0;
    selfattentionparam.inputLayout = atb::infer::TYPE_BSND;
    selfattentionparam.mlaVHeadSize = 0;
    selfattentionparam.windowSize = 0;

    std::vector<ms::Tensor> inputs, outputs{out};
    inputs = std::vector<ms::Tensor>{query, key, value, mask, seq_len};
    ms::pynative::RunAtbOp("FlashAttention", selfattentionparam, inputs, outputs);
}

auto pyboost_npu_flash_attention(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
    const ms::Tensor &mask, const ms::Tensor &seq_len, const double scale_value, const int64_t num_heads,
    const int64_t num_kv_heads, ms::Tensor out) {
  return ms::pynative::PyboostRunner::Call<0>(_npu_flash_attention, query, key, value, mask, seq_len, scale_value,
                                              num_heads, num_kv_heads, out);
}

void _npu_flash_attention_qlens(const ms::Tensor &query, const ms::Tensor &key_cache,
    const ms::Tensor &value_cache, const ms::Tensor &block_table, const ms::Tensor &mask, const ms::Tensor &seq_len,
    const ms::Tensor &context_lens, int64_t num_kv_heads, int64_t num_heads, double scale_value, ms::Tensor out)
{
    SelfAttentionParam selfattentionparam;

    selfattentionparam.calcType = SelfAttentionParam::PREFIX_ENCODER;
    selfattentionparam.kernelType = SelfAttentionParam::KERNELTYPE_HIGH_PRECISION;
    selfattentionparam.clampType = SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
    selfattentionparam.maskType = SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    selfattentionparam.kvcacheCfg = SelfAttentionParam::K_CACHE_V_CACHE;
    selfattentionparam.scaleType = SelfAttentionParam::SCALE_TYPE_TOR;
    selfattentionparam.quantType = SelfAttentionParam::TYPE_QUANT_UNQUANT;
    selfattentionparam.cacheType = SelfAttentionParam::CACHE_TYPE_NORM;
    selfattentionparam.outDataType = ACL_DT_UNDEFINED;
    selfattentionparam.headNum = num_heads;
    selfattentionparam.kvHeadNum = num_kv_heads;
    selfattentionparam.qScale = 1;
    selfattentionparam.qkScale = scale_value;
    selfattentionparam.batchRunStatusEnable = false;
    selfattentionparam.isTriuMask = 1;
    selfattentionparam.clampMin = 0;
    selfattentionparam.clampMax = 0;
    selfattentionparam.inputLayout = atb::infer::TYPE_BSND;
    selfattentionparam.mlaVHeadSize = 0;
    selfattentionparam.windowSize = 0;

    std::vector<ms::Tensor> inputs = {query, key_cache, value_cache, block_table, mask, seq_len, context_lens};
    ms::pynative::RunAtbOp("FlashAttention", selfattentionparam, inputs, {out});
}

auto pyboost_npu_flash_attention_qlens(const ms::Tensor &query, const ms::Tensor &key_cache,
    const ms::Tensor &value_cache, const ms::Tensor &block_table, const ms::Tensor &mask, const ms::Tensor &seq_len,
    const ms::Tensor &context_lens, int64_t num_kv_heads, int64_t num_heads, double scale_value, ms::Tensor out) {
  return ms::pynative::PyboostRunner::Call<0>(_npu_flash_attention_qlens, query, key_cache, value_cache, block_table,
                mask, seq_len, context_lens, num_kv_heads, num_heads, scale_value, out);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_selfattention", &pyboost_npu_selfattention, "SelfAttentionAtb", pybind11::arg("query"), pybind11::arg("key"),
        pybind11::arg("value"), pybind11::arg("mask"), pybind11::arg("seqLen"), pybind11::arg("scale"),
        pybind11::arg("headNum"), pybind11::arg("kvHeadNum"), pybind11::arg("out"), pybind11::arg("calcType"),
        pybind11::arg("kernelType"), pybind11::arg("clampType"), pybind11::arg("maskType"), pybind11::arg("kvcacheCfg"),
        pybind11::arg("scaleType"), pybind11::arg("quantType"), pybind11::arg("cacheType"),
        pybind11::arg("outDataType"), pybind11::arg("qScale"), pybind11::arg("batchRunStatusEnable"),
        pybind11::arg("isTriuMask"), pybind11::arg("clampMin"), pybind11::arg("clampMax"), pybind11::arg("inputLayout"),
        pybind11::arg("mlaVHeadSize"), pybind11::arg("windowSize"));

  m.def("npu_flash_attention", &pyboost_npu_flash_attention, "FlashAttentionAtb", pybind11::arg("query"), pybind11::arg("key"),
        pybind11::arg("value"), pybind11::arg("mask"), pybind11::arg("seq_len"), pybind11::arg("scale_value"),
        pybind11::arg("num_heads"), pybind11::arg("num_kv_heads"), pybind11::arg("out"));

  m.def("npu_flash_attention_qlens", &pyboost_npu_flash_attention_qlens, "FlashAttentionQlensAtb", pybind11::arg("query"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("block_table"), pybind11::arg("mask"), pybind11::arg("seq_len"), pybind11::arg("context_lens"),
        pybind11::arg("num_kv_heads"), pybind11::arg("num_heads"), pybind11::arg("scale_value"), pybind11::arg("out"));
}
