from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from configfuzz.gpu_campaign import (
    GPU_SUBJECTS,
    apply_frozen_feedback,
    build_frozen_gpu_targets,
    dump_frozen_gpu_targets,
    load_frozen_gpu_targets,
    summarize_frozen_gpu_results,
)
from configfuzz.dependencies import DependencyGraph, DependencyStatus
from configfuzz.model import Constraint, ConstraintKind, ConstraintSet
from configfuzz.outcomes import ClassifiedOutcome, OutcomeLabel, ProcessObservation
from configfuzz.probing import ProbeSample


ROOT = Path(__file__).resolve().parents[2]


def test_gpu_targets_freeze_deterministically(tmp_path: Path) -> None:
    first = build_frozen_gpu_targets(ROOT, limit_per_subject=2, solver_timeout_ms=2000)
    second = build_frozen_gpu_targets(ROOT, limit_per_subject=2, solver_timeout_ms=2000)

    assert first["frozen"] == second["frozen"]
    assert first["metadata"]["subject_count"] == len(GPU_SUBJECTS)
    assert first["metadata"]["target_count"] == sum(
        len(subject["targets"]) for subject in first["subjects"]
    )
    assert all(subject["targets"] for subject in first["subjects"])
    assert all(subject["harness_files"] for subject in first["subjects"])
    assert all(
        item["sha256"]
        for subject in first["subjects"]
        for item in subject["harness_files"]
    )

    output = tmp_path / "targets.yaml"
    dump_frozen_gpu_targets(first, output)
    loaded = load_frozen_gpu_targets(output)
    assert loaded["frozen"] == first["frozen"]


def test_expanded_megatron_targets_keep_qualified_process_topology() -> None:
    payload = build_frozen_gpu_targets(ROOT, limit_per_subject=20, solver_timeout_ms=5000)
    subject = next(item for item in payload["subjects"] if item["subject"] == "megatron-core")
    baseline = yaml.safe_load((ROOT / subject["baseline"]).read_text(encoding="utf-8"))

    assert len(subject["targets"]) == 11
    assert subject["selection_summary"]["pre_topology_filter_candidates"] == 19
    assert subject["selection_summary"]["skipped_unqualified_topology"] == 8
    topology = {
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
    }
    for target in subject["targets"]:
        cases = target["intervention"]["cases"]
        for role in ("satisfying", "repaired"):
            assignment = cases[role]["configuration"]
            for name in topology.intersection(assignment):
                assert assignment[name] == baseline[name]


def test_gpu_target_hash_detects_tampering(tmp_path: Path) -> None:
    payload = build_frozen_gpu_targets(ROOT, limit_per_subject=1, solver_timeout_ms=2000)
    tampered = deepcopy(payload)
    tampered["subjects"][0]["targets"][0]["expression"] += " and false"
    output = tmp_path / "tampered.yaml"
    output.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_gpu_targets(output)


def test_frozen_feedback_is_order_independent() -> None:
    constraints = (
        Constraint(expression="x > 1", kind=ConstraintKind.RELATION, parameters=("x",), confidence=0.8),
        Constraint(expression="x % 2 == 0", kind=ConstraintKind.RELATION, parameters=("x",), confidence=0.8),
    )
    item = ConstraintSet(parameter="x")
    for constraint in constraints:
        item.add(constraint)
    graph = DependencyGraph.from_constraint_sets([item])
    by_expression = {edge.expression: edge for edge in graph.edges.values()}

    def sample(edge_id: str, role: str, value: int, label: OutcomeLabel) -> ProbeSample:
        return ProbeSample(
            parameter="x",
            value=value,
            observation=ProcessObservation(("probe",), 0, "", "", 0.01),
            outcome=ClassifiedOutcome(label, "test"),
            intervention_id=f"pair-{edge_id}",
            intervention_edge_id=edge_id,
            intervention_role=role,
            provenance_matched=role == "violating",
        )

    first = by_expression["x > 1"].id
    second = by_expression["x % 2 == 0"].id
    groups = [
        [sample(first, "satisfying", 2, OutcomeLabel.VALID), sample(first, "violating", 1, OutcomeLabel.INVALID), sample(first, "repaired", 2, OutcomeLabel.VALID)],
        [sample(second, "satisfying", 2, OutcomeLabel.VALID), sample(second, "violating", 1, OutcomeLabel.INVALID), sample(second, "repaired", 2, OutcomeLabel.VALID)],
    ]

    forward_graph, forward_independent, forward_aggregate = apply_frozen_feedback(
        graph.to_dict(), {"x": 2}, groups
    )
    reverse_graph, reverse_independent, reverse_aggregate = apply_frozen_feedback(
        graph.to_dict(), {"x": 2}, list(reversed(groups))
    )

    assert forward_aggregate.paired_interventions == 2
    assert reverse_aggregate.paired_interventions == 2
    assert all(report.paired_interventions == 1 for report in forward_independent)
    assert all(report.paired_interventions == 1 for report in reverse_independent)
    assert {edge.status for edge in forward_graph.edges.values()} == {DependencyStatus.CONFIRMED}
    assert {edge.status for edge in reverse_graph.edges.values()} == {DependencyStatus.CONFIRMED}


def test_gpu_result_summary_is_derived_from_round_records() -> None:
    frozen = {
        "frozen": {"sha256": "frozen-sha", "target_count": 2},
        "subjects": [{"subject": "example"}],
    }
    results = {
        "example": {
            "subject": "example",
            "frozen_targets_sha256": "frozen-sha",
            "rounds": [
                {
                    "edge_id": "edge-confirmed",
                    "expression": "x > 0",
                    "feedback": {
                        "paired_interventions": 1,
                        "scope_disputed_edges": [],
                    },
                    "samples": [
                        {"outcome": {"label": "valid"}},
                        {"outcome": {"label": "invalid"}},
                        {"outcome": {"label": "valid"}},
                    ],
                },
                {
                    "edge_id": "edge-disputed",
                    "expression": "y > 0",
                    "feedback": {
                        "paired_interventions": 0,
                        "scope_disputed_edges": ["edge-disputed"],
                    },
                    "samples": [
                        {"outcome": {"label": "valid"}},
                        {"outcome": {"label": "valid"}},
                        {"outcome": {"label": "valid"}},
                    ],
                },
            ],
        }
    }

    summary = summarize_frozen_gpu_results(
        frozen,
        results,
        hardware={
            "accelerator": "GPU",
            "device_count": 2,
            "distributed_backend": "NCCL",
        },
        campaign_date="2026-08-15",
        runner_revision="revision",
        result_hashes={"example": "result-sha"},
    )

    assert summary["aggregate"] == {
        "targets": 2,
        "samples": 6,
        "paired_confirmed": 1,
        "scope_disputed": 1,
        "unresolved": 0,
        "outcomes": {"invalid": 1, "valid": 5},
    }
    assert summary["subjects"][0]["confirmed_edge_ids"] == ["edge-confirmed"]
    assert summary["subjects"][0]["scope_disputed_edge_ids"] == ["edge-disputed"]
    assert summary["subjects"][0]["result_sha256"] == "result-sha"
