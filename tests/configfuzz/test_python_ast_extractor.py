from __future__ import annotations

import ast
from pathlib import Path

from configfuzz.extractors.python_ast import (
    PythonConstraintExtractor,
    scan_python_paths_multi,
)
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


def test_does_not_promote_branch_local_alias_to_global_equivalence() -> None:
    result = extract(
        """
def prepare(config, enabled):
    tp = config.get("tensor_model_parallel_size", 1)
    if enabled:
        cp = tp

def validate(cp, heads):
    if heads % cp != 0:
        self._apply_fix("heads", heads, 8, "invalid")
""",
        "tensor_model_parallel_size",
    )

    assert expressions(result) == set()


def test_expands_local_symbolic_assignment_and_related_config_aliases() -> None:
    result = extract(
        """
def validate(config, heads):
    tp = config.get("tensor_model_parallel_size", 1)
    cp = config.get("context_parallel_size", 1)
    divisor = max(1, tp * cp)
    if heads % divisor != 0:
        self._apply_fix("heads", heads, 8, "invalid")
""",
        "tensor_model_parallel_size",
    )

    assert expressions(result) == {
        "heads % (tensor_model_parallel_size * context_parallel_size) == 0"
    }


def test_carries_enclosing_branch_into_conditional_constraint() -> None:
    result = extract(
        """
def validate(config):
    mode = config.get("router_mode")
    topk = config.get("moe_router_topk", 1)
    experts = config.get("num_experts", 1)
    if mode == "grouped":
        if topk > experts:
            self._apply_fix("moe_router_topk", topk, experts, "invalid")
""",
        "moe_router_topk",
    )

    assert expressions(result) == {
        "router_mode == 'grouped' => moe_router_topk <= num_experts"
    }
    assert result.constraints[0].kind is ConstraintKind.CONDITIONAL


def test_orients_boolean_requirement_toward_enabled_feature() -> None:
    result = extract(
        """
def validate(config):
    topk = config.get("moe_router_topk", 1)
    pre_softmax = config.get("moe_router_pre_softmax", False)
    if topk == 1 and not pre_softmax:
        self._apply_fix("moe_router_pre_softmax", False, True, "required")
""",
        "moe_router_topk",
    )

    assert expressions(result) == {
        "moe_router_topk == 1 => moe_router_pre_softmax"
    }


def test_ignores_two_sided_repair_decision() -> None:
    result = extract(
        """
def validate(config):
    tp = config.get("tensor_model_parallel_size", 1)
    if tp <= 4:
        self._apply_fix("pipeline_size", 1, 2, "first strategy")
    else:
        self._apply_fix("pipeline_size", 1, 1, "fallback strategy")
""",
        "tensor_model_parallel_size",
    )

    assert expressions(result) == set()


def test_ignores_computed_repair_change_detection() -> None:
    result = extract(
        """
def validate(config):
    batch = config.get("global_batch_size", 1)
    new_batch = nearest_valid_batch(batch)
    if new_batch != batch:
        self._apply_fix("global_batch_size", batch, new_batch, "adjust")
""",
        "global_batch_size",
    )

    assert expressions(result) == set()


def test_keeps_constant_repair_guard_as_constraint() -> None:
    result = extract(
        """
def validate(config):
    cp = config.get("context_parallel_size", 1)
    if cp != 1:
        self._apply_fix("context_parallel_size", cp, 1, "unsupported")
""",
        "context_parallel_size",
    )

    assert expressions(result) == {"context_parallel_size == 1"}


def test_strict_mode_filters_tensor_shape_and_path_checks() -> None:
    result = extract(
        """
def validate(hidden_size):
    if hidden_size.shape != (1, 2):
        raise ValueError("shape")
    if hidden_size.endswith(".json"):
        raise ValueError("path")
""",
        "hidden_size",
    )

    assert expressions(result) == set()
    assert result.metadata["filtered_candidates"] == 2


def test_multi_parameter_scan_parses_shared_file(tmp_path: Path) -> None:
    source = tmp_path / "validator.py"
    source.write_text(
        """
def validate(config):
    tp = config.get("tensor_model_parallel_size", 1)
    hidden = config.get("hidden_size", 1)
    if hidden % tp != 0:
        raise ValueError("invalid")
""",
        encoding="utf-8",
    )

    results = scan_python_paths_multi(
        [source],
        ["hidden_size", "tensor_model_parallel_size"],
    )

    expected = "hidden_size % tensor_model_parallel_size == 0"
    assert expected in expressions(results["hidden_size"])
    assert expected in expressions(results["tensor_model_parallel_size"])
    assert results["hidden_size"].metadata["scanned_files"] == 1


def test_parallel_scan_matches_serial_results(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    first.write_text(
        """
def validate(config):
    tp = config.get("tensor_model_parallel_size", 1)
    hidden = config.get("hidden_size", 1)
    if hidden % tp != 0:
        raise ValueError("invalid")
""",
        encoding="utf-8",
    )
    second = tmp_path / "second.py"
    second.write_text(
        """
def validate(hidden_size):
    if hidden_size < 1:
        raise ValueError("invalid")
""",
        encoding="utf-8",
    )
    parameters = ["hidden_size", "tensor_model_parallel_size"]

    serial = scan_python_paths_multi([tmp_path], parameters, jobs=1)
    parallel = scan_python_paths_multi([tmp_path], parameters, jobs=2)

    for parameter in parameters:
        assert expressions(serial[parameter]) == expressions(parallel[parameter])
        assert serial[parameter].metadata["scanned_files"] == parallel[parameter].metadata["scanned_files"]
