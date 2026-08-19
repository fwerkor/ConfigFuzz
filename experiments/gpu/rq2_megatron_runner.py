from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from rq2_family_factory import load_profile
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args as parse_megatron_args
from megatron.training.arguments import validate_args as validate_megatron_args
from runtime_events import RuntimeEventRecorder


MILESTONE_PREFIX = "CONFIGFUZZ_MILESTONE:"
SUPPORTED_FAMILIES = {"qwen2_dense_gqa", "llama2_dense_rope", "mixtral_moe"}


def milestone(name: str, rank: int = 0) -> None:
    if rank == 0:
        print(f"{MILESTONE_PREFIX}{name}", flush=True)


def _validate_native(profile: dict[str, object], world_size: int) -> None:
    model = profile["model"]
    training = profile["training"]
    parallel = profile["parallel"]
    argv = [
        "configfuzz-rq2-megatron",
        "--num-layers", str(model["num_layers"]),
        "--hidden-size", str(model["hidden_size"]),
        "--ffn-hidden-size", str(model["ffn_hidden_size"]),
        "--num-attention-heads", str(model["num_attention_heads"]),
        "--micro-batch-size", str(training["micro_batch_size"]),
        "--global-batch-size", str(training["global_batch_size"]),
        "--eval-global-batch-size", str(training["global_batch_size"]),
        "--eval-micro-batch-size", str(training["micro_batch_size"]),
        "--seq-length", str(model["seq_length"]),
        "--max-position-embeddings", str(model["max_position_embeddings"]),
        "--tensor-model-parallel-size", str(parallel.get("tensor_model_parallel_size", world_size)),
        "--pipeline-model-parallel-size", str(parallel.get("pipeline_model_parallel_size", 1)),
        "--context-parallel-size", str(parallel.get("context_parallel_size", 1)),
        "--expert-model-parallel-size", str(parallel.get("expert_model_parallel_size", 1)),
        "--distributed-backend", "nccl",
        "--normalization", "RMSNorm",
        "--disable-bias-linear",
    ]
    precision = profile.get("precision", {})
    if precision.get("bf16"):
        argv.append("--bf16")
    elif precision.get("fp16"):
        argv.append("--fp16")
    if parallel.get("sequence_parallel"):
        argv.append("--sequence-parallel")
    previous = sys.argv
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(io.StringIO()):
            native = parse_megatron_args()
            validate_megatron_args(native)
    except (AssertionError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"CONFIGFUZZ_NATIVE_VALIDATION_REJECTED: {type(exc).__name__}: {exc}") from exc
    finally:
        sys.argv = previous


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.config)
    if profile["family"] not in SUPPORTED_FAMILIES:
        raise ValueError(f"Megatron RQ2 has no native binding for {profile['family']}")
    milestone("argument_parsing")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    recorder = RuntimeEventRecorder(rank)
    recorder.emit_profile_state(profile)
    _validate_native(profile, world_size)
    milestone("configuration_validation", rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    model_cfg = profile["model"]
    parallel_cfg = profile["parallel"]
    tp = int(parallel_cfg.get("tensor_model_parallel_size", world_size))
    pp = int(parallel_cfg.get("pipeline_model_parallel_size", 1))
    cp = int(parallel_cfg.get("context_parallel_size", 1))
    ep = int(parallel_cfg.get("expert_model_parallel_size", 1))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=tp,
    )
    recorder.emit_distributed_state(
        framework="megatron_core",
        world_size=world_size,
        tp=parallel_state.get_tensor_model_parallel_world_size(),
        pp=parallel_state.get_pipeline_model_parallel_world_size(),
        cp=parallel_state.get_context_parallel_world_size(),
        ep=parallel_state.get_expert_model_parallel_world_size(),
    )
    recorder.emit("backend", "attention=megatron_local_spec")
    recorder.emit(
        "feature",
        f"sequence_parallel={'enabled' if bool(parallel_cfg.get('sequence_parallel', False)) else 'disabled'}",
    )
    seed = int(os.environ.get("CONFIGFUZZ_SEED", "2026"))
    model_parallel_cuda_manual_seed(seed)
    milestone("distributed_initialization", rank)

    precision = profile.get("precision", {})
    use_bf16 = bool(precision.get("bf16", False))
    use_fp16 = bool(precision.get("fp16", False))
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    moe = profile.get("moe", {})
    config = TransformerConfig(
        num_layers=int(model_cfg["num_layers"]),
        hidden_size=int(model_cfg["hidden_size"]),
        ffn_hidden_size=int(model_cfg["ffn_hidden_size"]),
        num_attention_heads=int(model_cfg["num_attention_heads"]),
        num_query_groups=int(model_cfg["num_query_groups"]),
        bf16=use_bf16,
        fp16=use_fp16,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=tp,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        normalization="RMSNorm",
        add_bias_linear=False,
        sequence_parallel=bool(parallel_cfg.get("sequence_parallel", False)),
        num_moe_experts=int(moe["num_experts"]) if moe else None,
        moe_router_topk=int(moe.get("moe_router_topk", 2)) if moe else 2,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(config.num_moe_experts, False),
        vocab_size=int(model_cfg["vocab_size"]),
        max_sequence_length=int(model_cfg["max_position_embeddings"]),
        parallel_output=False,
    ).to(device)
    if use_bf16 or use_fp16:
        model = Float16Module(config, model)
    recorder.instrument_model(model, profile)
    milestone("model_construction", rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(profile["training"]["learning_rate"]))
    batch_size = int(profile["training"]["micro_batch_size"])
    seq = int(model_cfg["seq_length"])
    vocab = int(model_cfg["vocab_size"])
    for step in range(int(profile["training"].get("train_iters", 2))):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + rank + step * world_size)
        tokens = torch.randint(0, vocab, (batch_size, seq + 1), generator=generator, device=device)
        input_ids, labels = tokens[:, :-1], tokens[:, 1:]
        position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.triu(torch.ones((1, 1, seq, seq), dtype=torch.bool, device=device), diagonal=1)
        optimizer.zero_grad(set_to_none=True)
        recorder.emit("branch", "forward_path=language")
        losses = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, labels=labels)
        loss = losses.float().mean()
        milestone("forward", rank)
        loss.backward()
        milestone("backward", rank)
        optimizer.step()
        milestone("optimizer_step", rank)
        if rank == 0:
            print(f"step={step} loss={loss.item():.6f}", flush=True)
    milestone("repeated_training", rank)

    if not args.skip_checkpoint:
        root = Path(os.environ.get("CONFIGFUZZ_CHECKPOINT_ROOT", "/tmp/configfuzz-rq2-megatron")) / str(profile["workload_id"])
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / f"rank-{rank}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        dist.barrier()
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        milestone("checkpoint_save_load", rank)
    milestone("completed", rank)
    recorder.close()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
