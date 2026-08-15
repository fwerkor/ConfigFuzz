from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist

from common import TinyCausalLM, batch, dtype_from_name, load_config, loss_from_logits, milestone


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    milestone("argument_parsing")

    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = TinyCausalLM(cfg).to(dtype=dtype_from_name(str(cfg.get("dtype", "bf16"))))
    milestone("model_construction", rank=rank)

    ds_cfg = {
        "train_micro_batch_size_per_gpu": int(
            cfg.get("train_micro_batch_size_per_gpu", cfg["micro_batch_size"])
        ),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "zero_optimization": {
            "stage": int(cfg.get("zero_stage", 1)),
            "reduce_bucket_size": int(cfg.get("reduce_bucket_size", 500_000_000)),
            "allgather_bucket_size": int(cfg.get("allgather_bucket_size", 500_000_000)),
            "overlap_comm": bool(cfg.get("overlap_comm", False)),
            "reduce_scatter": bool(cfg.get("reduce_scatter", True)),
        },
        "bf16": {"enabled": str(cfg.get("dtype", "bf16")).lower() in {"bf16", "bfloat16"}},
        "fp16": {"enabled": str(cfg.get("dtype", "bf16")).lower() in {"fp16", "float16"}},
        "steps_per_print": 1,
        "wall_clock_breakdown": False,
    }
    if cfg.get("train_batch_size") is not None:
        ds_cfg["train_batch_size"] = int(cfg["train_batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config=ds_cfg,
    )
    milestone("distributed_initialization", rank=rank)

    steps = int(cfg.get("qualification_steps", 2))
    for step in range(steps):
        input_ids, labels = batch(cfg, device)
        logits = engine(input_ids)
        loss = loss_from_logits(logits.float(), labels)
        milestone("forward", rank=rank)
        engine.backward(loss)
        milestone("backward", rank=rank)
        engine.step()
        milestone("optimizer_step", rank=rank)
        if rank == 0:
            print(f"step={step} loss={loss.item():.6f}", flush=True)
    milestone("repeated_training", rank=rank)

    checkpoint_dir_value = cfg.get("checkpoint_dir")
    if checkpoint_dir_value:
        checkpoint_dir = Path(checkpoint_dir_value)
        cleanup_checkpoint = False
    else:
        run_id = hashlib.sha256(str(Path(args.config).resolve()).encode()).hexdigest()[:16]
        checkpoint_dir = Path(tempfile.gettempdir()) / f"configfuzz-deepspeed-{run_id}"
        cleanup_checkpoint = True
        if rank == 0:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        if dist.is_initialized():
            dist.barrier()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(str(checkpoint_dir), tag="qualification")
    if dist.is_initialized():
        dist.barrier()
    engine.load_checkpoint(str(checkpoint_dir), tag="qualification")
    milestone("checkpoint_save_load", rank=rank)
    milestone("completed", rank=rank)
    if dist.is_initialized():
        dist.barrier()
    if cleanup_checkpoint and rank == 0:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
