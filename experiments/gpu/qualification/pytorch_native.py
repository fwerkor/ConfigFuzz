from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common import TinyCausalLM, batch, dtype_from_name, load_config, loss_from_logits, milestone


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    milestone("argument_parsing")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend=str(cfg.get("distributed_backend", "nccl")))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    milestone("distributed_initialization", rank=rank)

    model = TinyCausalLM(cfg)
    dtype = dtype_from_name(str(cfg.get("dtype", "bf16")))
    model = model.to(device=device, dtype=dtype)
    if bool(cfg.get("torch_compile", False)):
        model = torch.compile(model)
    milestone("model_construction", rank=rank)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))

    steps = int(cfg.get("qualification_steps", 2))
    for step in range(steps):
        input_ids, labels = batch(cfg, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = loss_from_logits(logits.float(), labels)
        milestone("forward", rank=rank)
        loss.backward()
        milestone("backward", rank=rank)
        optimizer.step()
        milestone("optimizer_step", rank=rank)
        if rank == 0:
            print(f"step={step} loss={loss.item():.6f}", flush=True)
    milestone("repeated_training", rank=rank)

    checkpoint_dir = Path(cfg.get("checkpoint_dir") or "artifacts/gpu/qualification/pytorch-native")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "qualification.pt"
    if rank == 0:
        module = model.module if isinstance(model, DDP) else model
        torch.save({"model": module.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    if world_size > 1:
        dist.barrier()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    module = model.module if isinstance(model, DDP) else model
    module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    milestone("checkpoint_save_load", rank=rank)
    milestone("completed", rank=rank)

    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
