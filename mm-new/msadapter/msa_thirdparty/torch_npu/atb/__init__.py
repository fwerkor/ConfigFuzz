import os
import sys
import numpy as np
import mindspore
from mindspore import ops

import torch

class AtbOpBuilder():
  def __init__(self):
    current_dir = os.path.dirname(__file__)
    csrc_root = f"{current_dir}/../../../msadapter/csrc"
    self.csrc_dir = f"{csrc_root}/atb_ops"
    self.atb_op = {}
  
  def load(self, csrc_name):
      if csrc_name not in self.atb_op:
        to_build = self.csrc_dir + "/" + csrc_name + ".cpp",
        atb_builder = ops.CustomOpBuilder(csrc_name + "_custom_atb", to_build, enable_atb=True)
        self.atb_op[csrc_name] = atb_builder.load()
      return self.atb_op[csrc_name]

builder = AtbOpBuilder()

def npu_reshapecache(key, value, keyCache, valueCache, slotMapping, compressType=0, kvCacheCfg=0):
  return builder.load('reshape_and_cache').npu_reshape_cache(key, value, keyCache, valueCache, slotMapping, compressType, kvCacheCfg)

def npu_pagedattention(
      query, keyCache, valueCache, contextLens, maskType, kvHeadNum, headNum, mlaVHeadSize, qkScale,
      scaleType, blockTables, batchRunStatusEnable, hasQuantOffset, calcType, quantType, compressType,
      inputLayout, outDataType, attnOut
    ):

    builder.load('paged_attention').npu_pagedattention(query, keyCache, valueCache, kvHeadNum, headNum, qkScale, blockTables,
        contextLens, attnOut, maskType, batchRunStatusEnable, quantType, outDataType, hasQuantOffset,
        compressType, calcType, scaleType, inputLayout, mlaVHeadSize
    )

def npu_selfattention(query, key, value, kvcacheCfg=0, mask=None, maskType=1, isTriuMask=0,
      seqLen=None, scale=1., qScale=1, scaleType=0, headNum=None, kvHeadNum=None,
      mlaVHeadSize=0, calcType=3, kernelType=0, clampType=0, quantType=0, cacheType=0, windowSize=0,
      clampMin=0, clampMax=0, batchRunStatusEnable=False, inputLayout=0, outDataType=0, out=None
    ):

    return builder.load('self_attention').npu_selfattention(
        query=query, key=key, value=value, kvcacheCfg=kvcacheCfg, mask=mask, maskType=maskType, isTriuMask=isTriuMask,
        seqLen=seqLen, scale=scale, qScale=qScale, scaleType=scaleType, headNum=headNum, kvHeadNum=kvHeadNum, 
        mlaVHeadSize=mlaVHeadSize, calcType=calcType, kernelType=kernelType, clampType=clampType, quantType=quantType, 
        cacheType=cacheType, windowSize=windowSize, clampMin=clampMin, clampMax=clampMax, batchRunStatusEnable=batchRunStatusEnable, 
        inputLayout=inputLayout, outDataType=outDataType, out=out)

def npu_group_topk(input, out, group_num, k):
    return builder.load('group_topk').npu_group_topk(input, k, group_num, 0, out)

def npu_matmul_add_fp32(x, weight, C):
    return builder.load("matmul_add").npu_matmul_add_fp32(x, weight, C)

# new atb operator
def _npu_reshape_and_cache(key, value, key_cache, value_cache, slot_indices):
  return builder.load('reshape_and_cache').npu_reshape_cache(key, value, key_cache, value_cache, slot_indices, 0, 0)

def _npu_reshape_and_cache_siso(key, key_cache, slot_indices):
  return builder.load('reshape_and_cache').npu_reshape_cache(key, None, key_cache, None, slot_indices, 0, 1)

def _npu_paged_attention(query, key_cache, value_cache, num_kv_heads, num_heads,
                         scale_value, block_table, context_lens, out):
    builder.load('paged_attention').npu_paged_attention(query, key_cache, value_cache, num_kv_heads, num_heads,
                                                               scale_value, block_table, context_lens, out)

def _npu_paged_attention_mla(query, key_cache, num_kv_heads, num_heads,
                             scale_value, block_table, context_lens, mla_vheadsize, out):
    builder.load('paged_attention').npu_paged_attention_mla(query, key_cache, num_kv_heads,
                                                                   num_heads, scale_value, block_table,
                                                                   context_lens, mla_vheadsize, out)

def _npu_paged_attention_splitfuse(query, key_cache, value_cache, block_table,
                                   context_lens, mask, seq_len, num_kv_heads, num_heads, scale_value, out):
    builder.load('paged_attention').npu_paged_attention_splitfuse(query, key_cache, value_cache,
                                                                  block_table, context_lens, mask, seq_len,
                                                                  num_kv_heads, num_heads, scale_value, out)

def _npu_rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style):
    builder.load('rope').npu_rope(positions, query, key, head_size, cos_sin_cache, is_neox_style)

def _npu_flash_attention(query, key, value, mask, seq_len, scale_value, num_heads, num_kv_heads, out):
    builder.load('self_attention').npu_flash_attention(query, key, value, mask, seq_len,
                                                              scale_value, num_heads, num_kv_heads, out)

def _npu_flash_attention_qlens(query, key_cache, value_cache, block_table, mask, seq_len,
                               context_lens, num_kv_heads, num_heads, scale_value, out):
    builder.load('self_attention').npu_flash_attention_qlens(query, key_cache, value_cache, block_table,
                                                                    mask, seq_len, context_lens, num_kv_heads,
                                                                    num_heads, scale_value, out)

def all_gather_matmul(input1, input2, bias, output, rank, tp_size, comm_domain):
    return builder.load("lcal_coc").all_gather_matmul(input1, input2, bias, output, rank, tp_size, comm_domain)
  
def all_gather_matmul_v2(input1, input2, bias, output, comm_output, rank, tp_size, comm_domain):
    return builder.load("lcal_coc").all_gather_matmul_v2(input1, input2, bias, output, comm_output, rank, tp_size, comm_domain)
    
def matmul_reduce_scatter(input1, input2, bias, output, rank, tp_size, comm_domain):
    return builder.load("lcal_coc").matmul_reduce_scatter(input1, input2, bias, output, rank, tp_size, comm_domain)
    
def matmul_all_reduce(input1, input2, bias, output, rank, tp_size, comm_domain):
    return builder.load("lcal_coc").matmul_all_reduce(input1, input2, bias, output, rank, tp_size, comm_domain)
    
def pure_matmul(input1, input2, bias, output, rank, tp_size, comm_domain):
    return builder.load("lcal_coc").pure_matmul(input1, input2, bias, output, rank, tp_size, comm_domain)

def npu_groupmatmul_add_fp32(x, weight, group_list, grad):
    if not isinstance(group_list, torch.Tensor):
        group_list = torch.tensor(group_list)
    return builder.load("groupmatmul_add").npu_groupmatmul_add_fp32(x, weight, group_list, grad)
