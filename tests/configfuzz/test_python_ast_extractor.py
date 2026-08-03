from __future__ import annotations

import ast

from configfuzz.extractors.python_ast import PythonConstraintExtractor
from configfuzz.model import ConstraintKind


def extract(source: str, parameter: str):
    tree = ast.parse(source)
    return PythonConstraintExtractor(parameter=parameter, source="example.py").extract(tree)


def expressions(result):
    return {constraint.expression for constraint in result.constraints}


def test_extracts_asserted_range_constraints() -> None:
    result = extract(
        """
def validate(batch_size):
    assert batch_size > 0 and batch_size <= 64
""",
        "batch_size",
    )

    assert expressions(result) == {"batch_size > 0", "batch_size <= 64"}
    assert {item.kind for item in result.constraints} == {ConstraintKind.RANGE}


def test_inverts_rejecting_or_guard() -> None:
    result = extract(
        """
def validate(sequence_length):
    if sequence_length < 1 or sequence_length > 8192:
        raise ValueError("invalid sequence length")
""",
        "sequence_length",
    )

    assert expressions(result) == {
        "sequence_length >= 1",
        "sequence_length <= 8192",
    }


def test_tracks_get_alias_and_lmsv_apply_fix() -> None:
    result = extract(
        """
def validate(config, hidden_size):
    parallel = config.get("parallel", {})
    tp = parallel.get("tensor_model_parallel_size", 1)
    if hidden_size % tp != 0:
        self._apply_fix("model.hidden_size", hidden_size, 4096, "must divide")
""",
        "tensor_model_parallel_size",
    )

    assert "hidden_size % tensor_model_parallel_size == 0" in expressions(result)
    constraint = result.constraints[0]
    assert constraint.kind is ConstraintKind.RELATION
    assert constraint.parameters[0] == "tensor_model_parallel_size"


def test_extracts_enumeration_constraint() -> None:
    result = extract(
        """
def validate(dtype):
    if dtype not in {"fp16", "bf16"}:
        raise ValueError(dtype)
""",
        "dtype",
    )

    assert expressions(result) == {"dtype in {'fp16', 'bf16'}"}
    assert result.constraints[0].kind is ConstraintKind.ENUM
