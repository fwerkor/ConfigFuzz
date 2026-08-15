from __future__ import annotations

import pytest

from configfuzz.dependencies import DependencyGraph, DependencyStatus
from configfuzz.graph_solver import SolveStatus, design_edge_intervention
from configfuzz.model import Constraint, ConstraintKind, ConstraintSet, Evidence, EvidenceKind


def make_graph(*constraints: Constraint) -> DependencyGraph:
    sets: dict[str, ConstraintSet] = {}
    for constraint in constraints:
        for parameter in constraint.parameters:
            sets.setdefault(parameter, ConstraintSet(parameter=parameter)).add(constraint)
    return DependencyGraph.from_constraint_sets(sets.values())


def constraint(
    expression: str,
    parameters: tuple[str, ...],
    *,
    kind: ConstraintKind = ConstraintKind.RELATION,
) -> Constraint:
    return Constraint(
        expression=expression,
        kind=kind,
        parameters=parameters,
        confidence=0.8,
        evidence=(
            Evidence(
                kind=EvidenceKind.STATIC,
                source="framework/config.py",
                line=12,
                detail="test evidence",
            ),
        ),
    )


def merged(base: dict[str, object], updates: dict[str, object]) -> dict[str, object]:
    return {**base, **updates}


def test_designs_minimal_divisibility_pair_and_repair() -> None:
    graph = make_graph(
        constraint("hidden_size % tensor_model_parallel_size == 0", ("hidden_size", "tensor_model_parallel_size")),
        constraint("hidden_size % 8 == 0", ("hidden_size",)),
    )
    target = next(
        edge
        for edge in graph.edges.values()
        if edge.expression == "hidden_size % tensor_model_parallel_size == 0"
    )
    baseline = {"hidden_size": 16, "tensor_model_parallel_size": 4}

    plan = design_edge_intervention(graph, baseline, target.id)

    assert plan.satisfying.status is SolveStatus.SAT
    assert plan.violating.status is SolveStatus.SAT
    assert plan.repaired is not None
    assert plan.repaired.status is SolveStatus.SAT
    satisfying = merged(baseline, plan.satisfying.assignment)
    violating = merged(baseline, plan.violating.assignment)
    repaired = merged(baseline, plan.repaired.assignment)
    assert graph.evaluate_edge(target, satisfying).satisfied is True
    assert graph.evaluate_edge(target, violating).satisfied is False
    assert graph.evaluate_edge(target, repaired).satisfied is True
    assert plan.primary_parameter == "hidden_size"
    assert plan.provenance[0]["source"] == "framework/config.py"


def test_guard_is_active_in_both_intervention_cases() -> None:
    graph = make_graph(
        constraint(
            "sequence_parallel => tensor_model_parallel_size > 1",
            ("sequence_parallel", "tensor_model_parallel_size"),
            kind=ConstraintKind.CONDITIONAL,
        )
    )
    edge = next(iter(graph.edges.values()))
    baseline = {"sequence_parallel": False, "tensor_model_parallel_size": 1}

    plan = design_edge_intervention(graph, baseline, edge.id)

    assert plan.satisfying.assignment["sequence_parallel"] is True
    assert plan.satisfying.assignment["tensor_model_parallel_size"] > 1
    assert plan.violating.assignment == {
        "sequence_parallel": True,
        "tensor_model_parallel_size": 1,
    }


def test_other_confirmed_edges_remain_hard() -> None:
    graph = make_graph(
        constraint("x % 2 == 0", ("x",)),
        constraint("x > 0", ("x",)),
    )
    target = next(edge for edge in graph.edges.values() if edge.expression == "x % 2 == 0")
    positive = next(edge for edge in graph.edges.values() if edge.expression == "x > 0")
    graph.update_status(positive.id, DependencyStatus.CONFIRMED)

    plan = design_edge_intervention(graph, {"x": 2}, target.id)

    assert plan.violating.status is SolveStatus.SAT
    assert plan.violating.assignment["x"] > 0
    assert plan.violating.assignment["x"] % 2 == 1
    assert positive.id in plan.violating.hard_edges


def test_reports_unsat_when_target_cannot_be_violated_without_breaking_hard_edge() -> None:
    graph = make_graph(
        constraint("x > 0", ("x",)),
        constraint("x >= 1", ("x",)),
    )
    target = next(edge for edge in graph.edges.values() if edge.expression == "x > 0")
    covering = next(edge for edge in graph.edges.values() if edge.expression == "x >= 1")
    graph.update_status(covering.id, DependencyStatus.CONFIRMED)

    plan = design_edge_intervention(graph, {"x": 1}, target.id)

    assert plan.satisfying.status is SolveStatus.SAT
    assert plan.violating.status is SolveStatus.UNSAT


def test_probe_templates_carry_intervention_metadata() -> None:
    graph = make_graph(constraint("x % 2 == 0", ("x",)))
    edge = next(iter(graph.edges.values()))

    payload = design_edge_intervention(graph, {"x": 2}, edge.id).to_dict()

    satisfying = payload["cases"]["satisfying"]["probe_sample_template"]
    violating = payload["cases"]["violating"]["probe_sample_template"]
    assert satisfying["intervention_id"] == payload["intervention_id"]
    assert satisfying["intervention_edge_id"] == edge.id
    assert satisfying["intervention_role"] == "satisfying"
    assert violating["intervention_role"] == "violating"


def test_unconfirmed_transitive_neighbors_do_not_expand_mutable_region() -> None:
    graph = make_graph(
        constraint("x % y == 0", ("x", "y")),
        constraint("y > 1 => z > 0", ("y", "z"), kind=ConstraintKind.CONDITIONAL),
    )
    target = next(edge for edge in graph.edges.values() if edge.expression == "x % y == 0")

    plan = design_edge_intervention(graph, {"x": 4, "y": 2}, target.id)

    assert plan.mutable_parameters == ("x", "y")
    assert plan.satisfying.status is SolveStatus.SAT
    assert plan.violating.status is SolveStatus.SAT


def test_intervention_timeout_must_be_positive() -> None:
    graph = make_graph(constraint("x % 2 == 0", ("x",)))
    edge = next(iter(graph.edges.values()))

    with pytest.raises(ValueError, match="timeout"):
        design_edge_intervention(graph, {"x": 2}, edge.id, timeout_ms=0)


def test_enum_repair_uses_baseline_as_deterministic_tie_break() -> None:
    graph = make_graph(
        constraint(
            "backend in {'nccl', 'gloo'}",
            ("backend",),
            kind=ConstraintKind.ENUM,
        )
    )
    edge = next(iter(graph.edges.values()))
    baseline = {"backend": "nccl"}

    first = design_edge_intervention(graph, baseline, edge.id)
    second = design_edge_intervention(graph, baseline, edge.id)

    assert first.violating.status is SolveStatus.SAT
    assert first.repaired is not None
    assert first.repaired.assignment == {"backend": "nccl"}
    assert second.repaired is not None
    assert second.repaired.assignment == first.repaired.assignment
