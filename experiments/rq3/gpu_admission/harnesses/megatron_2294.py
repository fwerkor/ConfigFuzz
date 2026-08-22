from __future__ import annotations

import os

import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    config = TransformerConfig(
        num_layers=5,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=3,
    )
    block = TransformerBlock(config, get_gpt_layer_with_transformer_engine_spec()).cuda()
    seq_len, micro_batch = 32, 2
    hidden_states = torch.ones(
        (seq_len, micro_batch, config.hidden_size), device="cuda", requires_grad=True
    )
    attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device="cuda")
    output = block(hidden_states=hidden_states, attention_mask=attention_mask)
    print(f"CONFIGFUZZ_RQ3 output_shape={tuple(output.shape)}", flush=True)
    if tuple(output.shape) != (seq_len, micro_batch, config.hidden_size):
        raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
    print("CONFIGFUZZ_RQ3_COMPLETED", flush=True)
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
