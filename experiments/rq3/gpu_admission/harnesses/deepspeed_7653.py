from __future__ import annotations

import os
import random

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RandomDataset(Dataset):
    def __init__(self, num_samples: int = 640, input_dim: int = 32, num_classes: int = 10):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return torch.randn(self.input_dim), torch.randint(0, self.num_classes, (1,)).item()


class RandomNet(nn.Module):
    def __init__(self, input_dim: int = 32, output_dim: int = 10):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(10):
            layers.extend((nn.Linear(dim, 512), nn.ReLU()))
            dim = 512
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        logits = self.net(x)
        return self.criterion(logits, labels) if labels is not None else logits


def main() -> None:
    set_seed()
    config = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 8,
        "optimizer": {"type": "Adam"},
        "communication_data_type": "bf16",
        "zero_optimization": {"stage": 1, "reduce_bucket_size": 10000},
    }
    model = RandomNet()
    data = RandomDataset()
    engine, _, loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=config,
        training_data=data,
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    for step, batch in enumerate(loader):
        x, y = batch
        x, y = x.to(engine.device), y.to(engine.device)
        loss = engine(x, labels=y)
        engine.backward(loss)
        engine.step()
        if rank == 0:
            print(f"CONFIGFUZZ_RQ3 step={step} loss={loss.item():.6f}", flush=True)
        if step >= 4:
            break
    if rank == 0:
        print("CONFIGFUZZ_RQ3_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
