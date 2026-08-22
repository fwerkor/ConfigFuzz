from __future__ import annotations

import math
import os

import torch
from torch.nn.functional import mse_loss

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    fully_shard_model,
    fully_shard_optimizer,
)

DIM_SIZE = 2
NUM_STEPS = 2


class RootParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(DIM_SIZE, DIM_SIZE))
        self.bias = torch.nn.Parameter(torch.empty(DIM_SIZE))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    torch.manual_seed(1234)
    reference_model = RootParamModel().cuda()
    model = RootParamModel().cuda()
    model.load_state_dict(reference_model.state_dict())
    model = fully_shard_model(
        module=model,
        fsdp_unit_modules=[RootParamModel],
        zero_dp_strategy="optim_grads_params",
    )

    reference_optimizer = torch.optim.AdamW(
        reference_model.parameters(), lr=0.01, weight_decay=0.1
    )
    optimizer = fully_shard_optimizer(
        torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    )

    data_generator = torch.Generator(device="cuda").manual_seed(
        91011 + torch.distributed.get_rank()
    )
    model_input = torch.randn(DIM_SIZE, DIM_SIZE, device="cuda", generator=data_generator)
    target = torch.randn(DIM_SIZE, DIM_SIZE, device="cuda", generator=data_generator)

    reference_losses = []
    losses = []
    for _ in range(NUM_STEPS):
        reference_optimizer.zero_grad()
        optimizer.zero_grad()
        reference_loss = mse_loss(reference_model(model_input), target)
        loss = mse_loss(model(model_input), target)
        reference_losses.append(reference_loss.detach())
        losses.append(loss.detach())
        reference_loss.backward()
        loss.backward()
        for param in reference_model.parameters():
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)
        reference_optimizer.step()
        optimizer.step()

    got = torch.stack(losses)
    expected = torch.stack(reference_losses)
    max_abs = float((got - expected).abs().max().cpu())
    print(f"CONFIGFUZZ_RQ3 losses={got.tolist()} reference={expected.tolist()} max_abs={max_abs:.9g}", flush=True)
    try:
        torch.testing.assert_close(got, expected)
    except AssertionError as exc:
        raise AssertionError("CONFIGFUZZ_RQ3_OPTIMIZER_UNDER_UPDATE") from exc
    print("CONFIGFUZZ_RQ3_COMPLETED", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
