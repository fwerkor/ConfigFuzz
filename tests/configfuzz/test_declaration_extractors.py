from __future__ import annotations

from pathlib import Path

from configfuzz.extractors import (
    scan_declaration_paths_multi,
    scan_source_paths_multi,
)
from configfuzz.model import ConstraintKind


def expressions(result):
    return {item.expression for item in result.constraints}


def test_extracts_argparse_type_choices_and_required(tmp_path: Path) -> None:
    source = tmp_path / "arguments.py"
    source.write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tensor-model-parallel-size",
    type=int,
    choices=[1, 2, 4, 8],
    required=True,
)
parser.add_argument("--sequence-parallel", action="store_true")
""",
        encoding="utf-8",
    )

    results = scan_declaration_paths_multi(
        [source],
        ["tensor_model_parallel_size", "sequence_parallel"],
    )

    assert expressions(results["tensor_model_parallel_size"]) == {
        "tensor_model_parallel_size: integer",
        "tensor_model_parallel_size in {1, 2, 4, 8}",
        "tensor_model_parallel_size is not None",
    }
    assert expressions(results["sequence_parallel"]) == {
        "sequence_parallel: boolean"
    }


def test_extracts_dataclass_literal_and_field_metadata(tmp_path: Path) -> None:
    source = tmp_path / "config.py"
    source.write_text(
        """
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class ModelConfig:
    hidden_size: int = field(metadata={"min": 128, "max": 16384, "multiple_of": 8})
    dtype: Literal["fp16", "bf16"] = "bf16"
    num_experts: Optional[int] = None
""",
        encoding="utf-8",
    )

    results = scan_declaration_paths_multi(
        [source],
        ["hidden_size", "dtype", "num_experts"],
    )

    assert expressions(results["hidden_size"]) == {
        "hidden_size: integer",
        "hidden_size is not None",
        "hidden_size >= 128",
        "hidden_size <= 16384",
        "hidden_size % 8 == 0",
    }
    assert expressions(results["dtype"]) == {
        'dtype in {"bf16", "fp16"}',
    }
    assert expressions(results["num_experts"]) == {
        "num_experts: integer",
    }


def test_extracts_json_schema_style_yaml(tmp_path: Path) -> None:
    schema = tmp_path / "schema.yaml"
    schema.write_text(
        """
type: object
properties:
  micro_batch_size:
    type: integer
    minimum: 1
    maximum: 64
  attention_impl:
    type: string
    enum: [eager, flash]
""",
        encoding="utf-8",
    )

    results = scan_declaration_paths_multi(
        [schema],
        ["micro_batch_size", "attention_impl"],
    )

    assert expressions(results["micro_batch_size"]) == {
        "micro_batch_size: integer",
        "micro_batch_size >= 1",
        "micro_batch_size <= 64",
    }
    assert expressions(results["attention_impl"]) == {
        "attention_impl: string",
        'attention_impl in {"eager", "flash"}',
    }


def test_extracts_existing_mutation_pool_shape(tmp_path: Path) -> None:
    pool = tmp_path / "mutable_params_pool.yaml"
    pool.write_text(
        """
numeric_params:
  attention_dropout:
    min_val: 0.0
    max_val: 1.0
enum_params:
  transformer_impl: [local, transformer_engine]
""",
        encoding="utf-8",
    )

    results = scan_declaration_paths_multi(
        [pool],
        ["attention_dropout", "transformer_impl"],
    )

    assert expressions(results["attention_dropout"]) == {
        "attention_dropout >= 0.0",
        "attention_dropout <= 1.0",
    }
    assert expressions(results["transformer_impl"]) == {
        'transformer_impl in {"local", "transformer_engine"}',
    }


def test_extracts_pydantic_style_field_constraints(tmp_path: Path) -> None:
    source = tmp_path / "deepspeed_config.py"
    source.write_text(
        """
from pydantic import BaseModel, Field

class ZeroConfig(BaseModel):
    stage: int = Field(default=0, ge=0, le=3)
    overlap_comm: bool = False
""",
        encoding="utf-8",
    )

    results = scan_declaration_paths_multi(
        [source],
        ["stage", "overlap_comm"],
    )

    assert expressions(results["stage"]) == {
        "stage: integer",
        "stage >= 0",
        "stage <= 3",
    }
    assert expressions(results["overlap_comm"]) == {
        "overlap_comm: boolean",
    }


def test_extracts_json_schema_declarations(tmp_path: Path) -> None:
    schema = tmp_path / "config.schema.json"
    schema.write_text(
        """{
  "type": "object",
  "properties": {
    "train_batch_size": {"type": "integer", "minimum": 1},
    "mixed_precision": {"type": "string", "enum": ["no", "fp16", "bf16"]}
  }
}
""",
        encoding="utf-8",
    )

    results = scan_declaration_paths_multi(
        [schema],
        ["train_batch_size", "mixed_precision"],
    )

    assert expressions(results["train_batch_size"]) == {
        "train_batch_size: integer",
        "train_batch_size >= 1",
    }
    assert expressions(results["mixed_precision"]) == {
        "mixed_precision: string",
        'mixed_precision in {"bf16", "fp16", "no"}',
    }
    assert results["train_batch_size"].metadata["json_files"] == 1


def test_combined_scan_merges_declarations_and_runtime_guards(tmp_path: Path) -> None:
    source = tmp_path / "framework.py"
    source.write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hidden-size", type=int)

def validate(hidden_size, tp):
    if hidden_size % tp != 0:
        raise ValueError("invalid hidden size")
""",
        encoding="utf-8",
    )

    result = scan_source_paths_multi(
        [source],
        ["hidden_size"],
        jobs=1,
    )["hidden_size"]

    assert expressions(result) == {
        "hidden_size: integer",
        "hidden_size % tp == 0",
    }
    kinds = {item.expression: item.kind for item in result.constraints}
    assert kinds["hidden_size: integer"] is ConstraintKind.TYPE
    assert kinds["hidden_size % tp == 0"] is ConstraintKind.RELATION
