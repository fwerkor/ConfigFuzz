from __future__ import annotations

import os

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 0},
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=None,
        config=config,
        dist_init_required=False,
    )
    print(
        f"CONFIGFUZZ_RQ3 optimizer={type(engine.optimizer).__name__} "
        f"using_real_optimizer={engine.optimizer.using_real_optimizer} "
        f"bf16_groups={len(engine.optimizer.bf16_groups)}",
        flush=True,
    )
    engine.destroy()
    print("CONFIGFUZZ_RQ3_COMPLETED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
