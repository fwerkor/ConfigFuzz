from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from rq2_family_factory import build_model, load_profile


MILESTONE_PREFIX = "CONFIGFUZZ_MILESTONE:"


def milestone(name: str, rank: int = 0) -> None:
    if rank == 0:
        print(f"{MILESTONE_PREFIX}{name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Architecture-faithful reduced RQ2 GPU runner.")
    parser.add_argument("--framework", choices=("pytorch", "deepspeed", "accelerate"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()

    profile = load_profile(args.config)
    milestone("argument_parsing")
    if args.framework == "pytorch":
        return _run_pytorch(profile, args)
    if args.framework == "deepspeed":
        return _run_deepspeed(profile, args)
    return _run_accelerate(profile, args)


def _run_pytorch(profile, args) -> int:
    rank, local_rank, world_size, device = _torch_distributed_context()
    milestone("distributed_initialization", rank)
    model = build_model(profile).to(device=device, dtype=_dtype(profile))
    milestone("model_construction", rank)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate(profile))
    _train_steps(profile, model, optimizer, device, rank, backward=lambda loss: loss.backward())
    if not args.skip_checkpoint:
        checkpoint = _checkpoint_dir(profile, "pytorch") / "state.pt"
        if rank == 0:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            module = model.module if isinstance(model, DDP) else model
            torch.save({"model": module.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        if dist.is_initialized():
            dist.barrier()
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        module = model.module if isinstance(model, DDP) else model
        module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        milestone("checkpoint_save_load", rank)
    milestone("completed", rank)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


def _run_deepspeed(profile, args) -> int:
    import deepspeed

    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = build_model(profile).to(dtype=_dtype(profile))
    milestone("model_construction", rank)
    training = profile["training"]
    ds = profile.get("framework", {}).get("deepspeed", {})
    ds_config = {
        "train_micro_batch_size_per_gpu": int(ds.get("train_micro_batch_size_per_gpu", training["micro_batch_size"])),
        "gradient_accumulation_steps": int(ds.get("gradient_accumulation_steps", training.get("gradient_accumulation_steps", 1))),
        "zero_optimization": {
            "stage": int(ds.get("stage", 1)),
            "reduce_bucket_size": int(ds.get("reduce_bucket_size", 50_000_000)),
            "allgather_bucket_size": int(ds.get("allgather_bucket_size", 50_000_000)),
            "overlap_comm": bool(ds.get("overlap_comm", False)),
            "reduce_scatter": bool(ds.get("reduce_scatter", True)),
        },
        "bf16": {"enabled": _dtype(profile) is torch.bfloat16},
        "fp16": {"enabled": _dtype(profile) is torch.float16},
        "steps_per_print": 1,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate(profile))
    engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), optimizer=optimizer, config=ds_config)
    device = engine.device
    milestone("distributed_initialization", rank)
    _train_steps(profile, engine, optimizer, device, rank, backward=engine.backward, step=engine.step)
    if not args.skip_checkpoint:
        checkpoint = _checkpoint_dir(profile, "deepspeed")
        checkpoint.mkdir(parents=True, exist_ok=True)
        engine.save_checkpoint(str(checkpoint), tag="formal-preflight")
        if dist.is_initialized():
            dist.barrier()
        engine.load_checkpoint(str(checkpoint), tag="formal-preflight")
        milestone("checkpoint_save_load", rank)
    milestone("completed", rank)
    if dist.is_initialized():
        dist.barrier()
    return 0


def _run_accelerate(profile, args) -> int:
    from accelerate import Accelerator

    framework = profile.get("framework", {}).get("accelerate", {})
    accelerator = Accelerator(
        mixed_precision=str(framework.get("mixed_precision", "bf16")),
        gradient_accumulation_steps=int(profile["training"].get("gradient_accumulation_steps", 1)),
    )
    rank = accelerator.process_index
    milestone("distributed_initialization", rank)
    model = build_model(profile)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate(profile))
    model, optimizer = accelerator.prepare(model, optimizer)
    milestone("model_construction", rank)
    _train_steps(profile, model, optimizer, accelerator.device, rank, backward=accelerator.backward)
    if not args.skip_checkpoint:
        checkpoint = _checkpoint_dir(profile, "accelerate")
        checkpoint.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(checkpoint))
        accelerator.wait_for_everyone()
        accelerator.load_state(str(checkpoint))
        milestone("checkpoint_save_load", rank)
    milestone("completed", rank)
    accelerator.wait_for_everyone()
    return 0


def _train_steps(profile, model, optimizer, device, rank, *, backward, step=None) -> None:
    steps = int(profile["training"].get("train_iters", 2))
    seed = int(os.environ.get("CONFIGFUZZ_SEED", "2026"))
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _forward_loss(profile, model, device, rank, index, seed)
        milestone("forward", rank)
        backward(loss)
        milestone("backward", rank)
        if step is None:
            optimizer.step()
        else:
            step()
        milestone("optimizer_step", rank)
        if rank == 0:
            print(f"step={index} loss={float(loss.detach()):.6f}", flush=True)
    milestone("repeated_training", rank)


def _forward_loss(profile, model, device, rank: int, step: int, seed: int = 2026) -> torch.Tensor:
    family = str(profile["family"])
    if family == "cogvideox_video_text":
        return _cogvideox_loss(profile, model, device, rank, step, seed)
    return _language_or_multimodal_loss(profile, model, device, rank, step, seed)


def _language_or_multimodal_loss(
    profile, model, device, rank: int, step: int, seed: int = 2026
) -> torch.Tensor:
    training = profile["training"]
    model_cfg = profile["model"]
    batch_size = int(training["micro_batch_size"])
    seq = int(model_cfg["seq_length"])
    vocab = int(model_cfg["vocab_size"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rank + step * 997)
    tokens = torch.randint(2, vocab, (batch_size, seq), generator=generator, device=device)
    labels = tokens.clone()
    kwargs = {"input_ids": tokens, "labels": labels, "use_cache": False}
    if str(profile["family"]) == "internvl3_vision_text":
        mm = profile["multimodal"]
        image_tokens = int(mm["image_seq_length"])
        if image_tokens >= seq:
            raise ValueError("InternVL image token count must be smaller than sequence length")
        tokens[:, :image_tokens] = 1
        labels = tokens.clone()
        pixels = torch.randn(
            batch_size,
            int(mm["num_channels"]),
            int(mm["image_size"]),
            int(mm["image_size"]),
            generator=generator,
            device=device,
            dtype=_dtype(profile),
        )
        kwargs.update(input_ids=tokens, labels=labels, pixel_values=pixels)
    outputs = model(**kwargs)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError("model did not return a training loss")
    return loss


def _cogvideox_loss(
    profile, model, device, rank: int, step: int, seed: int = 2026
) -> torch.Tensor:
    video = profile["video"]
    batch_size = int(profile["training"]["micro_batch_size"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rank + step * 997)
    hidden_states = torch.randn(
        batch_size,
        int(video["frames"]),
        int(video["in_channels"]),
        int(video["sample_height"]),
        int(video["sample_width"]),
        generator=generator,
        device=device,
        dtype=_dtype(profile),
    )
    encoder_hidden_states = torch.randn(
        batch_size,
        int(video["max_text_seq_length"]),
        int(video["text_embed_dim"]),
        generator=generator,
        device=device,
        dtype=_dtype(profile),
    )
    timestep = torch.full((batch_size,), step + 1, device=device, dtype=torch.long)
    outputs = model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep)
    sample = outputs.sample if hasattr(outputs, "sample") else outputs[0]
    return sample.float().square().mean()


def _torch_distributed_context():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _dtype(profile) -> torch.dtype:
    precision = profile.get("precision", {})
    bf16 = bool(precision.get("bf16", False))
    fp16 = bool(precision.get("fp16", False))
    if bf16 and fp16:
        raise ValueError("bf16 and fp16 are mutually exclusive")
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.float32


def _learning_rate(profile) -> float:
    return float(profile["training"].get("learning_rate", 1e-4))


def _checkpoint_dir(profile, framework: str) -> Path:
    raw = os.environ.get("CONFIGFUZZ_CHECKPOINT_ROOT")
    if raw:
        base = Path(raw)
    else:
        digest = hashlib.sha256(str(profile["workload_id"]).encode()).hexdigest()[:10]
        base = Path(tempfile.gettempdir()) / f"configfuzz-rq2-{digest}"
    return base / framework / str(profile["workload_id"])


if __name__ == "__main__":
    raise SystemExit(main())
