from dataclasses import dataclass, field
from typing import Literal

from checks import require_divisible

@dataclass
class TransformerConfig:
    hidden_size: int = field(metadata={"minimum": 128, "multiple_of": 8})
    tensor_model_parallel_size: int = 1
    dtype: Literal["fp16", "bf16"] = "bf16"


def validate(config):
    require_divisible(config.hidden_size, config.tensor_model_parallel_size)
