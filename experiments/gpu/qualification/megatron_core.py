from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import io
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from common import load_config, milestone
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args as parse_megatron_args
from megatron.training.arguments import validate_args as validate_megatron_args


def _validate_native_training_args(cfg: dict[str, object]) -> None:
    """Run Megatron's training argument validator on the effective test configuration."""
    argv = [
        "configfuzz-megatron-validation",
        "--num-layers",
        str(cfg["num_layers"]),
        "--hidden-size",
        str(cfg["hidden_size"]),
        "--ffn-hidden-size",
        str(cfg["ffn_hidden_size"]),
        "--num-attention-heads",
        str(cfg["num_attention_heads"]),
        "--micro-batch-size",
        str(cfg["micro_batch_size"]),
        "--global-batch-size",
        str(cfg["global_batch_size"]),
        "--eval-global-batch-size",
        str(cfg.get("eval_global_batch_size", cfg["global_batch_size"])),
        "--eval-micro-batch-size",
        str(cfg.get("eval_micro_batch_size", cfg["micro_batch_size"])),
        "--seq-length",
        str(cfg["sequence_length"]),
        "--max-position-embeddings",
        str(cfg["max_position_embeddings"]),
        "--tensor-model-parallel-size",
        str(cfg.get("tensor_model_parallel_size", 1)),
        "--pipeline-model-parallel-size",
        str(cfg.get("pipeline_model_parallel_size", 1)),
        "--context-parallel-size",
        str(cfg.get("context_parallel_size", 1)),
        "--expert-model-parallel-size",
        str(cfg.get("expert_model_parallel_size", 1)),
        "--distributed-backend",
        "nccl",
        "--normalization",
        str(cfg.get("normalization", "RMSNorm")),
        "--transformer-impl",
        str(cfg.get("transformer_impl", "transformer_engine")),
    ]
    if bool(cfg.get("sequence_parallel", False)):
        argv.append("--sequence-parallel")
    if bool(cfg.get("fp16", False)):
        argv.append("--fp16")
    if bool(cfg.get("bf16", False)):
        argv.append("--bf16")
    if not bool(cfg.get("add_bias_linear", False)):
        argv.append("--disable-bias-linear")

    previous_argv = sys.argv
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(io.StringIO()):
            native_args = parse_megatron_args()
            validate_megatron_args(native_args)
    except (AssertionError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"CONFIGFUZZ_NATIVE_VALIDATION_REJECTED: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.argv = previous_argv



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    milestone("argument_parsing")
    _validate_native_training_args(cfg)
    milestone("configuration_validation")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    tensor_parallel_size = int(cfg.get("tensor_model_parallel_size", world_size))
    pipeline_parallel_size = int(cfg.get("pipeline_model_parallel_size", 1))
    context_parallel_size = int(cfg.get("context_parallel_size", 1))
    expert_parallel_size = int(cfg.get("expert_model_parallel_size", 1))
    expert_tensor_parallel_size = int(
        cfg.get("expert_tensor_parallel_size", tensor_parallel_size)
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
    )
    model_parallel_cuda_manual_seed(int(cfg.get("seed", 2026)))
    milestone("distributed_initialization", rank=rank)

    use_bf16 = bool(cfg.get("bf16", False))
    use_fp16 = bool(cfg.get("fp16", False))
    params_dtype = (
        torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    )
    transformer_impl = str(cfg.get("transformer_impl", "local"))
    normalization_impl = str(cfg.get("normalization_impl", "local"))
    if transformer_impl not in {"local", "transformer_engine"}:
        raise ValueError(f"unsupported transformer_impl for GPU qualification: {transformer_impl}")
    if normalization_impl not in {"local", "transformer_engine"}:
        raise ValueError(
            f"unsupported normalization_impl for GPU qualification: {normalization_impl}"
        )
    config = TransformerConfig(
        num_layers=int(cfg["num_layers"]),
        transformer_impl=transformer_impl,
        hidden_size=int(cfg["hidden_size"]),
        ffn_hidden_size=int(cfg["ffn_hidden_size"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        num_query_groups=int(cfg.get("num_query_groups", cfg["num_attention_heads"])),
        bf16=use_bf16,
        fp16=use_fp16,
        sequence_parallel=bool(cfg.get("sequence_parallel", False)),
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,
        normalization=str(cfg.get("normalization", "RMSNorm")),
        add_bias_linear=bool(cfg.get("add_bias_linear", False)),
        num_moe_experts=(
            int(cfg["num_moe_experts"])
            if cfg.get("num_moe_experts") is not None
            else None
        ),
    )
    if transformer_impl == "transformer_engine":
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
    else:
        local_spec = get_gpt_layer_local_spec(
            normalization=str(cfg.get("normalization", "RMSNorm"))
        )
        if normalization_impl == "transformer_engine":
            layer_spec = replace(
                local_spec,
                submodules=replace(
                    local_spec.submodules,
                    input_layernorm=TENorm,
                    pre_mlp_layernorm=TENorm,
                ),
            )
        else:
            layer_spec = local_spec
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=int(cfg["vocab_size"]),
        max_sequence_length=int(cfg["max_position_embeddings"]),
        parallel_output=False,
    ).to(device)
    if use_fp16 or use_bf16:
        model = Float16Module(config, model)
    milestone("model_construction", rank=rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    batch_size = int(cfg["micro_batch_size"])
    seq = int(cfg["sequence_length"])
    vocab = int(cfg["vocab_size"])
    steps = int(cfg.get("qualification_steps", 2))

    for step in range(steps):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(cfg.get("seed", 2026)) + rank + step * world_size)
        tokens = torch.randint(0, vocab, (batch_size, seq + 1), generator=generator, device=device)
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.triu(
            torch.ones((1, 1, seq, seq), dtype=torch.bool, device=device), diagonal=1
        )
        optimizer.zero_grad(set_to_none=True)
        losses = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = losses.float().mean()
        milestone("forward", rank=rank)
        loss.backward()
        milestone("backward", rank=rank)
        optimizer.step()
        milestone("optimizer_step", rank=rank)
        if rank == 0:
            print(f"step={step} loss={loss.item():.6f}", flush=True)
    milestone("repeated_training", rank=rank)

    checkpoint_dir = Path(cfg.get("checkpoint_dir") or "artifacts/gpu/qualification/megatron-core")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"rank-{rank}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    dist.barrier()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    milestone("checkpoint_save_load", rank=rank)
    milestone("completed", rank=rank)

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
