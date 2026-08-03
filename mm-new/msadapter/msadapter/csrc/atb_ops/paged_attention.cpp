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
struct HashOpParam<atb::infer::PagedAttentionParam> {
  void operator()(const atb::infer::PagedAttentionParam &param) const {
    add_param_to_buf("kvHeadNum", param.kvHeadNum);
    add_param_to_buf("headNum", param.headNum);
    add_param_to_buf("qkScale", param.qkScale);
    add_param_to_buf("quantType", param.quantType);
    add_param_to_buf("outDataType", param.outDataType);
    add_param_to_buf("mlaVHeadSize", param.mlaVHeadSize);
    add_param_to_buf("maskType", param.maskType);
    add_param_to_buf("batchRunStatusEnable", param.batchRunStatusEnable);
    add_param_to_buf("hasQuantOffset", param.hasQuantOffset);
    add_param_to_buf("compressType", param.compressType);
    add_param_to_buf("calcType", param.calcType);
    add_param_to_buf("scaleType", param.scaleType);
    add_param_to_buf("inputLayout", param.inputLayout);
  }
};
}  // namespace atb

class AtbOpWithMixDevice : public ms::pynative::AtbOpRunner {
 public:
  using AtbOpRunner::AtbOpRunner;
  void SetAscendTensors(const std::vector<ms::Tensor> &inputs) { ascend_inputs_ = inputs; }
  void _PrepareDeviceAddress() override {
    _inputs_.swap(ascend_inputs_);
    AtbOpRunner::_PrepareDeviceAddress();
  }
  size_t CalcWorkspace() override {
    _inputs_.swap(ascend_inputs_);
    return AtbOpRunner::CalcWorkspace();
  }

 protected:
  std::vector<ms::Tensor> ascend_inputs_;
};

using PagedAttentionParam = atb::infer::PagedAttentionParam;

void npu_pagedattention(const ms::Tensor &query, const ms::Tensor &keyCache,
                        const std::optional<ms::Tensor> &valueCache, int64_t kvHeadNum, int64_t headNum,
                        double qkScale, const ms::Tensor &blockTables, const ms::Tensor &contextLens,
                        ms::Tensor attnOut, int64_t maskType, bool batchRunStatusEnable, int64_t quantType,
                        int64_t outDataType, bool hasQuantOffset, int64_t compressType, int64_t calcType,
                        int64_t scaleType, int64_t inputLayout, int64_t mlaVHeadSize) {
  PagedAttentionParam pagedparam;
  pagedparam.headNum = headNum;
  pagedparam.qkScale = qkScale;
  pagedparam.kvHeadNum = kvHeadNum;
  auto masktype = static_cast<PagedAttentionParam::MaskType>(maskType);
  pagedparam.maskType = masktype;
  pagedparam.batchRunStatusEnable = batchRunStatusEnable;
  auto quanttype = static_cast<PagedAttentionParam::QuantType>(quantType);
  pagedparam.quantType = quanttype;
  auto outdataType = static_cast<aclDataType>(outDataType);
  pagedparam.outDataType = outdataType;
  pagedparam.hasQuantOffset = hasQuantOffset;
  auto compresstype = static_cast<PagedAttentionParam::CompressType>(compressType);
  pagedparam.compressType = compresstype;
  auto calctype = static_cast<PagedAttentionParam::CalcType>(calcType);
  pagedparam.calcType = calctype;
  auto scaletype = static_cast<PagedAttentionParam::ScaleType>(scaleType);
  pagedparam.scaleType = scaletype;
  auto inputlayout = static_cast<atb::infer::InputLayout>(inputLayout);
  pagedparam.inputLayout = inputlayout;
  auto mlavHeadSize = static_cast<uint32_t>(mlaVHeadSize);
  pagedparam.mlaVHeadSize = mlavHeadSize;

  std::vector<ms::Tensor> inputs;
  std::vector<ms::Tensor> inputs_ascend;
  if (pagedparam.mlaVHeadSize > 0) {
    inputs = std::vector<ms::Tensor>{query, keyCache, blockTables, contextLens};
    inputs_ascend = std::vector<ms::Tensor>{query, keyCache, blockTables};
  } else {
    ms::Tensor value_t = valueCache.has_value() ? valueCache.value() : ms::Tensor();
    inputs = std::vector<ms::Tensor>{query, keyCache, value_t, blockTables, contextLens};
    inputs_ascend = std::vector<ms::Tensor>{query, keyCache, value_t, blockTables};
  }
  auto runner = std::make_shared<AtbOpWithMixDevice>("PagedAttention");
  runner->Init(pagedparam);
  runner->SetAscendTensors(inputs_ascend);
  runner->Run(inputs, {attnOut});
}

void _npu_paged_attention(const ms::Tensor &query, const ms::Tensor &key_cache, const ms::Tensor &value_cache,
                          int64_t num_kv_heads, int64_t num_heads, double scale_value, const ms::Tensor &block_table,
                          const ms::Tensor &context_lens, ms::Tensor out) {
  PagedAttentionParam pagedparam;
  pagedparam.headNum = num_heads;
  pagedparam.qkScale = scale_value;
  pagedparam.kvHeadNum = num_kv_heads;
  pagedparam.maskType = PagedAttentionParam::UNDEFINED;
  pagedparam.batchRunStatusEnable = false;
  pagedparam.quantType = PagedAttentionParam::TYPE_QUANT_UNDEFINED;
  pagedparam.outDataType = ACL_DT_UNDEFINED;
  pagedparam.hasQuantOffset = false;
  pagedparam.compressType = PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
  pagedparam.calcType = PagedAttentionParam::CALC_TYPE_UNDEFINED;
  pagedparam.scaleType = PagedAttentionParam::SCALE_TYPE_TOR;
  pagedparam.inputLayout = atb::infer::TYPE_BSND;
  pagedparam.mlaVHeadSize = 0;

  std::vector<ms::Tensor> inputs = {query, key_cache, value_cache, block_table, context_lens};
  std::vector<ms::Tensor> inputs_ascend = {query, key_cache, value_cache, block_table};
  auto runner = std::make_shared<AtbOpWithMixDevice>("PagedAttentionOperation");
  runner->Init(pagedparam);
  runner->SetAscendTensors(inputs_ascend);
  runner->Run(inputs, {out});
  return;
}

void _npu_paged_attention_mla(const ms::Tensor &query, const ms::Tensor &key_cache, int64_t num_kv_heads,
                              int64_t num_heads, double scale_value, const ms::Tensor &block_table,
                              const ms::Tensor &context_lens, int64_t mla_vheadsize, ms::Tensor out) {
  PagedAttentionParam pagedparam;
  pagedparam.headNum = num_heads;
  pagedparam.qkScale = scale_value;
  pagedparam.kvHeadNum = num_kv_heads;
  auto mlavHeadSize = static_cast<uint32_t>(mla_vheadsize);
  pagedparam.mlaVHeadSize = mlavHeadSize;
  pagedparam.maskType = PagedAttentionParam::UNDEFINED;
  pagedparam.batchRunStatusEnable = false;
  pagedparam.quantType = PagedAttentionParam::TYPE_QUANT_UNDEFINED;
  pagedparam.outDataType = ACL_DT_UNDEFINED;
  pagedparam.hasQuantOffset = false;
  pagedparam.compressType = PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
  pagedparam.calcType = PagedAttentionParam::CALC_TYPE_UNDEFINED;
  pagedparam.scaleType = PagedAttentionParam::SCALE_TYPE_TOR;
  pagedparam.inputLayout = atb::infer::TYPE_BSND;

  std::vector<ms::Tensor> inputs = {query, key_cache, block_table, context_lens};
  std::vector<ms::Tensor> inputs_ascend = {query, key_cache, block_table};
  auto runner = std::make_shared<AtbOpWithMixDevice>("PagedAttention_mla");
  runner->Init(pagedparam);
  runner->SetAscendTensors(inputs_ascend);
  runner->Run(inputs, {out});
  return;
}

void _npu_paged_attention_splitfuse(const ms::Tensor &query, const ms::Tensor &key_cache,
                                    const ms::Tensor &value_cache, const ms::Tensor &block_table,
                                    const ms::Tensor &context_lens, const ms::Tensor &mask,
                                    const ms::Tensor &seq_len, int64_t num_kv_heads, int64_t num_heads,
                                    double scale_value, ms::Tensor out) {
  PagedAttentionParam pagedparam;
  pagedparam.headNum = num_heads;
  pagedparam.qkScale = scale_value;
  pagedparam.kvHeadNum = num_kv_heads;
  pagedparam.maskType = PagedAttentionParam::MASK_TYPE_SPEC;
  pagedparam.batchRunStatusEnable = false;
  pagedparam.quantType = PagedAttentionParam::TYPE_QUANT_UNDEFINED;
  pagedparam.outDataType = ACL_DT_UNDEFINED;
  pagedparam.hasQuantOffset = false;
  pagedparam.compressType = PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
  pagedparam.calcType = PagedAttentionParam::CALC_TYPE_SPEC;
  pagedparam.scaleType = PagedAttentionParam::SCALE_TYPE_TOR;
  pagedparam.inputLayout = atb::infer::TYPE_BSND;
  pagedparam.mlaVHeadSize = 0;

  std::vector<ms::Tensor> inputs = {query, key_cache, value_cache, block_table, context_lens, mask, seq_len};
  std::vector<ms::Tensor> inputs_ascend = {query, key_cache, value_cache, block_table, mask, seq_len};
  auto runner = std::make_shared<AtbOpWithMixDevice>("PagedAttention_splitfuse");
  runner->Init(pagedparam);
  runner->SetAscendTensors(inputs_ascend);
  runner->Run(inputs, {out});
  return;
}

auto pyboost_npu_pagedattention(const ms::Tensor &query, const ms::Tensor &keyCache,
                        const std::optional<ms::Tensor> &valueCache, int64_t kvHeadNum, int64_t headNum,
                        double qkScale, const ms::Tensor &blockTables, const ms::Tensor &contextLens,
                        ms::Tensor attnOut, int64_t maskType, bool batchRunStatusEnable, int64_t quantType,
                        int64_t outDataType, bool hasQuantOffset, int64_t compressType, int64_t calcType,
                        int64_t scaleType, int64_t inputLayout, int64_t mlaVHeadSize) {
  return ms::pynative::PyboostRunner::Call<0>(npu_pagedattention, query, keyCache, valueCache,
      kvHeadNum, headNum, qkScale, blockTables, contextLens, attnOut, maskType, batchRunStatusEnable,
      quantType, outDataType, hasQuantOffset, compressType, calcType, scaleType, inputLayout, mlaVHeadSize);
}

auto pyboost_npu_paged_attention(const ms::Tensor &query, const ms::Tensor &key_cache, const ms::Tensor &value_cache,
                          int64_t num_kv_heads, int64_t num_heads, double scale_value, const ms::Tensor &block_table,
                          const ms::Tensor &context_lens, ms::Tensor out) {
return ms::pynative::PyboostRunner::Call<0>(_npu_paged_attention, query, key_cache, value_cache, num_kv_heads, num_heads,
      scale_value, block_table, context_lens, out);
}

auto pyboost_npu_paged_attention_mla(const ms::Tensor &query, const ms::Tensor &key_cache, int64_t num_kv_heads,
                              int64_t num_heads, double scale_value, const ms::Tensor &block_table,
                              const ms::Tensor &context_lens, int64_t mla_vheadsize, ms::Tensor out) {
  return ms::pynative::PyboostRunner::Call<0>(_npu_paged_attention_mla, query, key_cache, num_kv_heads, num_heads,
      scale_value, block_table, context_lens, mla_vheadsize, out);
}

auto pyboost_npu_paged_attention_splitfuse(const ms::Tensor &query, const ms::Tensor &key_cache,
                                    const ms::Tensor &value_cache, const ms::Tensor &block_table,
                                    const ms::Tensor &context_lens, const ms::Tensor &mask,
                                    const ms::Tensor &seq_len, int64_t num_kv_heads, int64_t num_heads,
                                    double scale_value, ms::Tensor out) {
  return ms::pynative::PyboostRunner::Call<0>(_npu_paged_attention_splitfuse, query, key_cache, value_cache,
      block_table, context_lens, mask, seq_len, num_kv_heads, num_heads, scale_value, out);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_pagedattention", &pyboost_npu_pagedattention, "PagedAttention", pybind11::arg("query"), pybind11::arg("keyCache"),
        pybind11::arg("valueCache"), pybind11::arg("kvHeadNum"), pybind11::arg("headNum"), pybind11::arg("qkScale"),
        pybind11::arg("blockTables"), pybind11::arg("contextLens"), pybind11::arg("attnOut"), pybind11::arg("maskType"),
        pybind11::arg("batchRunStatusEnable"), pybind11::arg("quantType"), pybind11::arg("outDataType"),
        pybind11::arg("hasQuantOffset"), pybind11::arg("compressType"), pybind11::arg("calcType"),
        pybind11::arg("scaleType"), pybind11::arg("inputLayout"), pybind11::arg("mlaVHeadSize"));

  m.def("npu_paged_attention", &pyboost_npu_paged_attention, "PagedAttentionAtb", pybind11::arg("query"),
        pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("num_kv_heads"),
        pybind11::arg("num_heads"), pybind11::arg("scale_value"), pybind11::arg("block_table"),
        pybind11::arg("context_lens"), pybind11::arg("out"));

  m.def("npu_paged_attention_mla", &pyboost_npu_paged_attention_mla, "PagedAttentionMlaAtb", pybind11::arg("query"),
        pybind11::arg("key_cache"), pybind11::arg("num_kv_heads"), pybind11::arg("num_heads"),
        pybind11::arg("scale_value"), pybind11::arg("block_table"), pybind11::arg("context_lens"),
        pybind11::arg("mla_vheadsize"), pybind11::arg("out"));

  m.def("npu_paged_attention_splitfuse", &pyboost_npu_paged_attention_splitfuse, "PagedAttentionSplitfuseAtb",
        pybind11::arg("query"), pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("block_table"),
        pybind11::arg("context_lens"), pybind11::arg("mask"), pybind11::arg("seq_len"), pybind11::arg("num_kv_heads"),
        pybind11::arg("num_heads"), pybind11::arg("scale_value"), pybind11::arg("out"));
}
