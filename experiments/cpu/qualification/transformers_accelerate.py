from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import LlamaConfig, LlamaForCausalLM

from common import batch, load_config, milestone


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    milestone("argument_parsing")

    mixed_precision = str(cfg.get("mixed_precision", "no"))
    accelerator = Accelerator(
        cpu=True,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
    )
    rank = accelerator.process_index
    milestone("distributed_initialization", rank=rank)

    model_cfg = LlamaConfig(
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(cfg["ffn_hidden_size"]),
        num_hidden_layers=int(cfg["num_layers"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        num_key_value_heads=int(cfg.get("num_query_groups", cfg["num_attention_heads"])),
        max_position_embeddings=int(cfg["max_position_embeddings"]),
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    model = LlamaForCausalLM(model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("learning_rate", 1e-3))
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    milestone("model_construction", rank=rank)

    for step in range(int(cfg.get("qualification_steps", 2))):
        input_ids, labels = batch(cfg, accelerator.device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=labels).loss
        milestone("forward", rank=rank)
        accelerator.backward(loss)
        milestone("backward", rank=rank)
        optimizer.step()
        milestone("optimizer_step", rank=rank)
        if accelerator.is_main_process:
            print(f"step={step} loss={loss.item():.6f}", flush=True)
    milestone("repeated_training", rank=rank)

    checkpoint_dir = Path(
        cfg.get("checkpoint_dir")
        or "artifacts/cpu/qualification/transformers-accelerate"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()
    accelerator.load_state(str(checkpoint_dir))
    milestone("checkpoint_save_load", rank=rank)
    milestone("completed", rank=rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
