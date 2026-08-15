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

    output = tmp_path / "targets.yaml"
    dump_frozen_gpu_targets(first, output)
    loaded = load_frozen_gpu_targets(output)
    assert loaded["frozen"] == first["frozen"]


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
