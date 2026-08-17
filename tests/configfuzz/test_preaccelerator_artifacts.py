from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_canonical_rq2_workloads_are_balanced_and_preflighted() -> None:
    registry = _yaml("experiments/rq2/canonical_workloads.prequalified.yaml")
    assert len(registry["workloads"]) == 7
    for entry in registry["workloads"]:
        profile = _json("experiments/rq2/" + entry["baseline_config"])
        assert profile["model_scale_profile"] == "npu-reduced-26.1"
        assert profile["model"]["num_layers"] == 4
        assert profile["model"]["hidden_size"] == 512
        assert profile["model"]["ffn_hidden_size"] == 1024
        assert profile["model"]["seq_length"] == 128

    preflight = _json("experiments/rq2/model_preflight.prequalified.json")
    assert preflight["passed"] == 7
    assert preflight["failed"] == 0
    assert {row["status"] for row in preflight["records"]} == {"model_forward_backward_passed"}


def test_rq2_matrix_intents_and_schedule_are_frozen() -> None:
    matrix = _yaml("experiments/rq2/framework_workload_matrix.yaml")
    assert len({row["framework_id"] for row in matrix["bindings"]}) == 6
    assert sum(bool(row["formal_rq2"]) for row in matrix["bindings"]) == 38

    intents = _yaml("experiments/rq2/intents.prequalified.frozen.yaml")
    rows = intents["intents"]
    counts = Counter(row["workload_id"] for row in rows)
    assert len(rows) == 1050
    assert len(counts) == 7
    assert set(counts.values()) == {150}

    schedule = _yaml("experiments/rq2/replication_summary.prequalified.yaml")
    assert schedule["total_record_count"] == 58560
    assert len(schedule["subjects"]) == 6


def test_rq3_history_is_fully_prepared_for_accelerator_replay() -> None:
    bindings = _yaml("experiments/rq3/search_bindings.prequalified.yaml")
    assert bindings["metadata"]["source_commit_candidates"] == 23
    assert bindings["metadata"]["logical_root_causes"] == 25
    assert len(bindings["cases"]) == 25

    preflight = _json("experiments/rq3/history_preflight.prequalified.json")
    assert preflight["logical_case_count"] == 25
    assert preflight["ready_count"] == 25
    assert preflight["failed_count"] == 0
