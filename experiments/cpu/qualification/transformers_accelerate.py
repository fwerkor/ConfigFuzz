from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.commands.config import get_config_parser
from accelerate.commands.config.default import write_basic_config
from accelerate.commands.launch import launch_command_parser
from transformers import LlamaConfig, LlamaForCausalLM

from common import batch, load_config, milestone


_WRITE_BASIC_CONFIG_EDGE = "dep-39fd2282ed77229c"
_DEFAULT_CONFIG_CLI_EDGE = "dep-5d2bdf54140b8f5a"
_LAUNCH_CLI_EDGE = "dep-bb93e90843aba2d0"


def _parse_with_provenance(parser, argv: list[str], source: str):
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code:
            print(f"CONFIGFUZZ_PROVENANCE:{source}", file=sys.stderr, flush=True)
        raise


def validate_recovered_mixed_precision(cfg: dict) -> None:
    edge_id = os.environ.get("CONFIGFUZZ_INTERVENTION_EDGE", "")
    mixed_precision = str(cfg.get("mixed_precision", "no"))
    if edge_id == _WRITE_BASIC_CONFIG_EDGE:
        with tempfile.TemporaryDirectory(prefix="configfuzz-accelerate-config-") as raw_temp:
            write_basic_config(
                mixed_precision,
                save_location=str(Path(raw_temp) / "default_config.yaml"),
            )
    elif edge_id == _DEFAULT_CONFIG_CLI_EDGE:
        with tempfile.TemporaryDirectory(prefix="configfuzz-accelerate-config-") as raw_temp:
            _parse_with_provenance(
                get_config_parser(),
                [
                    "default",
                    "--mixed_precision",
                    mixed_precision,
                    "--config_file",
                    str(Path(raw_temp) / "default_config.yaml"),
                ],
                "src/accelerate/commands/config/default.py",
            )
    elif edge_id == _LAUNCH_CLI_EDGE:
        _parse_with_provenance(
            launch_command_parser(),
            [
                "--cpu",
                "--mixed_precision",
                mixed_precision,
                "--num_processes",
                "2",
                "configfuzz_cpu_probe.py",
            ],
            "src/accelerate/commands/launch.py",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    milestone("argument_parsing")

    validate_recovered_mixed_precision(cfg)
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
