from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


MILESTONE_PREFIX = "CONFIGFUZZ_MILESTONE:"


def load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a JSON object")
    return payload


def milestone(name: str, *, rank: int = 0) -> None:
    if rank == 0:
        print(f"{MILESTONE_PREFIX}{name}", flush=True)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


class TinyCausalLM(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = int(config["hidden_size"])
        heads = int(config["num_attention_heads"])
        ffn = int(config["ffn_hidden_size"])
        layers = int(config["num_layers"])
        vocab = int(config["vocab_size"])
        max_seq = int(config["max_position_embeddings"])
        self.token_embedding = nn.Embedding(vocab, hidden)
        self.position_embedding = nn.Embedding(max_seq, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq = input_ids.shape[1]
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq, device=input_ids.device)
        hidden = self.layers(hidden, mask=mask, is_causal=True)
        return self.lm_head(self.norm(hidden))


def batch(config: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(config["micro_batch_size"])
    seq = int(config["sequence_length"])
    vocab = int(config["vocab_size"])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.get("seed", 1)) + int(os.environ.get("RANK", "0")))
    tokens = torch.randint(0, vocab, (batch_size, seq + 1), generator=generator, device=device)
    return tokens[:, :-1], tokens[:, 1:]


def loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
