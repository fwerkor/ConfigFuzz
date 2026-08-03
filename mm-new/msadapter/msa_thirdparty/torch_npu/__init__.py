from . import npu
from . import _C
from . import profiler
from .atb import npu_reshapecache, npu_pagedattention, npu_selfattention, npu_group_topk, npu_matmul_add_fp32, \
                _npu_reshape_and_cache, _npu_reshape_and_cache_siso, _npu_paged_attention, _npu_paged_attention_mla, \
                _npu_paged_attention_splitfuse, _npu_rotary_embedding, _npu_flash_attention, _npu_flash_attention_qlens, \
                all_gather_matmul, all_gather_matmul_v2, matmul_reduce_scatter, matmul_all_reduce, pure_matmul, \
                npu_groupmatmul_add_fp32
from .cann import npu_apply_fused_ema_adamw
import torch
from torch.nn.functional import fast_gelu, swiglu
import mindspore as ms
from mindspore import ops
from mindspore.ops import auto_generate as gen
from mindspore import default_generator

unsupported_dtype = []

def npu_moe_init_routing(x, row_idx, expert_idx, active_num):
    moe_init_routing = gen.MoeInitRouting()
    return moe_init_routing(x, row_idx, expert_idx, active_num)

def npu_moe_compute_expert_tokens(sorted_experts, num_expert):
    moe_compute_expert_tokens = gen.MoeComputeExpertTokens()
    return moe_compute_expert_tokens(sorted_experts, num_expert)

def npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row,      
            export_for_source_row, drop_pad_mode=0):
    if skip1 is not None and bias is not None and scales is not None and \
            export_for_source_row is not None and drop_pad_mode == 0:
        return gen.MoeFinalizeRouting()(expanded_permuted_rows, skip1, skip2, bias, scales,
    expanded_src_to_dst_row, export_for_source_row)
    else:
        raise("MoeFinalizeRoutingV2 is not currently supported!")

def npu_grouped_matmul(x, weight, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None,
                    per_token_scale=None, group_list=None, activation_input=None, activation_quant_scale=None,
                    activation_quant_offset=None, split_item=0, group_type=None, group_list_type=0, act_type=0,
                    output_dtype=None):
    if split_item == 1:
        split_item = 0
    elif split_item == 2:
        split_item = 3
    group_list.data_sync(True)
    return gen.grouped_matmul_v4(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale,
                        group_list, activation_input, activation_quant_scale, activation_quant_offset, split_item=split_item, 
                        group_type=group_type, group_list_type=group_list_type, act_type=act_type)


def npu_rms_norm(x, gamma, epsilon=1e-5):
    output = ops.rms_norm(x, gamma, epsilon)
    return output


def npu_swiglu(x, dim=-1):
    return swiglu(x, dim)


from mindspore.ops import rotary_position_embedding

def npu_add_rms_norm(x1, x2, gamma, epsilon=1e-6):
    return ops.add_rms_norm(x1, x2, gamma, epsilon)

def npu_rotary_mul(x, cos, sin, rotary_mode="half"):
    mode = 0
    if rotary_mode == "interleave":
        mode = 1
    return rotary_position_embedding(x, cos, sin, mode=mode)


def npu_rotary_position_embedding(x, cos, sin, mode=0):
    """
    Inputs:
        - **x** (Tensor) - The input tensor.
        - **cos** (Tensor) - The input cos tensor.
        - **sin** (Tensor) - The input sin tensor.
        - **mode** (int) - Optional mode value: 0 rotate half, 1 rotate interleaved.

    Outputs:
        - **y** (Tensor) - The output tensor.

    """
    return rotary_position_embedding(x, cos, sin, mode)


def npu_fusion_attention(query, key, value, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                         scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0,
                         drop_mask=None, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                         gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None):
    output = gen.flash_attention_score_impl(
        query, key, value, real_shift=pse, padding_mask=padding_mask, drop_mask=drop_mask,
        attn_mask=atten_mask, prefix=prefix, actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen, head_num=head_num, keep_prob=float(keep_prob),
        scale_value=scale, pre_tokens=pre_tockens, next_tokens=next_tockens,
        inner_precise=inner_precise, input_layout=input_layout, sparse_mode=sparse_mode
    )
    sfm_max, sfm_sum, sfm_out, atten_out = output

    return atten_out, sfm_max, sfm_sum


def npu_fusion_attention_grad(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                            drop_mask=None, softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None, prefix=None,
                            actual_seq_qlen=None, actual_seq_kvlen=None, keep_prob=1.0, scale_value=1.0,
                            pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, sparse_mode=0,
                            gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None,
                            seed=1234, offset=0, numels=0):
    output = gen.flash_attention_score_grad_impl(query, key, value, dy, pse_shift=pse, padding_mask=padding_mask, atten_mask=atten_mask, drop_mask=drop_mask,
                                            softmax_max=softmax_max, softmax_sum=softmax_sum, softmax_in=softmax_in, attention_in=attention_in, prefix=prefix,
                                            actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, head_num=head_num, keep_prob=keep_prob,
                                            scale_value=scale_value, pre_tokens=pre_tockens, next_tokens=next_tockens, inner_precise=inner_precise,
                                            input_layout=input_layout, sparse_mode=sparse_mode)
    dq, dk, dv, dpse = output

    return dq, dk, dv

adamw_opt = gen.ApplyAdamW()


def npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2,
                     epsilon, grad, max_grad_norm, amsgrad, maximize, out):
    var, m, v = out
    var, m, v = adamw_opt(var, m, v, beta1_power=beta1_power, beta2_power=beta2_power, lr=lr, weight_decay=weight_decay,
                          beta1=beta1, beta2=beta2, epsilon=epsilon, grad=grad, max_grad_norm=max_grad_norm,
                          amsgrad=amsgrad, maximize=maximize)
    return var, m, v


adamw_opt_aclnn = gen.AdamW()

def npu_apply_adam_w_aclnn(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, 
                           state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize):
        
    var, m, v = adamw_opt_aclnn(var=param, m=exp_avg, v=exp_avg_sq, max_v=max_exp_avg_sq, 
                                gradient=grad, step=state_step, lr=lr, beta1=beta1, beta2=beta2, 
                                decay=weight_decay, eps=eps, amsgrad=amsgrad, maximize=maximize)
    return var, m, v


def npu_all_gather_base_mm(
        input_: ms.Tensor,
        x2: ms.Tensor,
        group: str,
        world_size: int,
        bias: None = None,
        gather_index: int = 0,
        gather_output: bool = True,
        comm_turn: int = 0,
) -> None:
    return ms.ops.all_gather_matmul(
        input_,
        x2,
        group,
        world_size,
        bias=bias,
        gather_index=gather_index,
        gather_output=gather_output,
        comm_turn=comm_turn,
    )


def npu_mm_reduce_scatter_base(
        input_: ms.Tensor,
        x2: ms.Tensor,
        group: str,
        world_size: int,
        reduce_op: str = ops.ReduceOp.SUM,
        bias: None = None,
        comm_turn: int = 0,
) -> None:
    return ms.ops.matmul_reduce_scatter(
        input_,
        x2,
        group,
        world_size,
        reduce_op=reduce_op,
        bias=bias,
        comm_turn=comm_turn,
    )


def npu_incre_flash_attention(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, 
                              actual_seq_lengths=None, antiquant_scale=None, antiquant_offset=None, block_table=None, 
                              dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, 
                              quant_offset2=None, kv_padding_size=None, num_heads=1, scale_value=1.0, 
                              input_layout="BSH", num_key_value_heads=0, block_size=0, inner_precise=1):
    key = [key]
    value = [value]
    output = ops.incre_flash_attention(
        query, key, value, attn_mask=atten_mask, actual_seq_lengths=actual_seq_lengths, pse_shift=pse_shift, 
        dequant_scale1=dequant_scale1, quant_scale1=quant_scale1, dequant_scale2=dequant_scale2, 
        quant_scale2=quant_scale2, quant_offset2=quant_offset2, antiquant_scale=antiquant_scale, 
        antiquant_offset=antiquant_offset, block_table=block_table, num_heads=num_heads, input_layout=input_layout, 
        scale_value=scale_value, num_key_value_heads=num_key_value_heads, block_size=block_size, 
        inner_precise=inner_precise, kv_padding_size=kv_padding_size
    )
    return output


def npu_prompt_flash_attention(query, key, value, *, pse_shift=None, padding_mask=None, atten_mask=None, 
                               actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, 
                               quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0, 
                               pre_tokens=2147473647, next_tokens=0, input_layout="BSH", num_key_value_heads=0, 
                               actual_seq_lengths_kv=None, sparse_mode=0):
    
    output = ops.prompt_flash_attention(
        query, key, value, attn_mask=atten_mask, actual_seq_lengths=actual_seq_lengths, 
        actual_seq_lengths_kv=actual_seq_lengths_kv, pse_shift=pse_shift, deq_scale1=deq_scale1, 
        quant_scale1=quant_scale1, deq_scale2=deq_scale2, quant_scale2=quant_scale2, quant_offset2=quant_offset2, 
        num_heads=num_heads, scale_value=scale_value, pre_tokens=pre_tokens, next_tokens=next_tokens, 
        input_layout=input_layout, num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode, inner_precise=1
    )
    return output

def npu_format_cast(tensor, acl_format):
    ...

def npu_dropout_gen_mask(size, p, device):
    mask_tensor = ms.ops.zeros(size, dtype=ms.float32)
    mask_tensor.uniform_(from_=0.0, to=1.0, generator=default_generator)
    mask = (mask_tensor < p).astype(ms.float32)
    return mask


def npu_fused_infer_attention_score(query, key, value, *, pse_shift=None, atten_mask=None, actual_seq_lengths=None,
                                    actual_seq_lengths_kv=None, dequant_scale1=None, quant_scale1=None,
                                    dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None,
                                    antiquant_offset=None, block_table=None, query_padding_size=None,
                                    kv_padding_size=None, key_antiquant_scale=None, key_antiquant_offset=None,
                                    value_antiquant_scale=None, value_antiquant_offset=None, key_shared_prefix=None,
                                    value_shared_prefix=None, actual_shared_prefix_len=None, num_heads=1, scale=1.0,
                                    pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH",
                                    num_key_value_heads=0, sparse_mode=0, inner_precise=0, block_size=0,
                                    antiquant_mode=0, softmax_lse_flag=False, key_antiquant_mode=0,
                                    value_antiquant_mode=0):
    out = ops.fused_infer_attention_score(query, key, value, pse_shift=pse_shift, atten_mask=atten_mask,
                                          actual_seq_lengths=actual_seq_lengths,
                                          actual_seq_lengths_kv=actual_seq_lengths_kv, dequant_scale1=dequant_scale1,
                                          quant_scale1=quant_scale1, dequant_scale2=dequant_scale2,
                                          quant_scale2=quant_scale2, quant_offset2=quant_offset2,
                                          antiquant_scale=antiquant_scale, antiquant_offset=antiquant_offset,
                                          key_antiquant_scale=key_antiquant_scale,
                                          key_antiquant_offset=key_antiquant_offset,
                                          value_antiquant_scale=value_antiquant_scale,
                                          value_antiquant_offset=value_antiquant_offset,
                                          block_table=block_table, query_padding_size=query_padding_size,
                                          kv_padding_size=kv_padding_size, key_shared_prefix=key_shared_prefix,
                                          value_shared_prefix=value_shared_prefix,
                                          actual_shared_prefix_len=actual_shared_prefix_len, num_heads=num_heads,
                                          scale=scale, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                          input_layout=input_layout, num_key_value_heads=num_key_value_heads,
                                          sparse_mode=sparse_mode, inner_precise=inner_precise, block_size=block_size,
                                          antiquant_mode=antiquant_mode, key_antiquant_mode=key_antiquant_mode,
                                          value_antiquant_mode=value_antiquant_mode, softmax_lse_flag=softmax_lse_flag)
    return out

def empty_with_swapped_memory(size:int, dtype=torch.float32, device=None):
    cpu_tensor = torch.empty(size, dtype=dtype, device="CPU")
    return cpu_tensor._shared_host_memory_with_device_()