from __future__ import annotations

from configfuzz.dependencies import DependencyGraph, DependencyStatus
from configfuzz.model import Constraint, ConstraintKind, ConstraintSet
from configfuzz.selection import select_interventions


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
    confidence: float = 0.6,
) -> Constraint:
    return Constraint(
        expression=expression,
        kind=kind,
        parameters=parameters,
        confidence=confidence,
    )


def test_selects_only_edges_with_satisfying_and_violating_cases() -> None:
    graph = make_graph(
        constraint("x % y == 0", ("x", "y")),
        constraint("x: integer", ("x",), kind=ConstraintKind.TYPE),
    )

    queue = select_interventions(graph, {"x": 4, "y": 2})

    assert [item.expression for item in queue.candidates] == ["x % y == 0"]
    assert queue.skipped_infeasible == 1


def test_confirmed_and_contradicted_edges_are_not_selected() -> None:
    graph = make_graph(
        constraint("x % 2 == 0", ("x",)),
        constraint("x > 0", ("x",)),
        constraint("x < 100", ("x",)),
    )
    by_expression = {edge.expression: edge for edge in graph.edges.values()}
    graph.update_status(by_expression["x > 0"].id, DependencyStatus.CONFIRMED)
    graph.update_status(by_expression["x < 100"].id, DependencyStatus.CONTRADICTED)

    queue = select_interventions(graph, {"x": 2})

    assert [item.expression for item in queue.candidates] == ["x % 2 == 0"]
    assert queue.skipped_status == 2


def test_scope_disputed_edge_is_prioritized_over_supported_peer() -> None:
    graph = make_graph(
        constraint("x % 2 == 0", ("x",), confidence=0.7),
        constraint("y % 2 == 0", ("y",), confidence=0.7),
    )
    by_expression = {edge.expression: edge for edge in graph.edges.values()}
    graph.update_status(
        by_expression["x % 2 == 0"].id,
        DependencyStatus.SCOPE_DISPUTED,
    )
    graph.update_status(
        by_expression["y % 2 == 0"].id,
        DependencyStatus.DYNAMICALLY_SUPPORTED,
    )

    queue = select_interventions(graph, {"x": 2, "y": 2})

    assert queue.candidates[0].expression == "x % 2 == 0"
    assert queue.candidates[0].score > queue.candidates[1].score


def test_conditional_high_order_relation_exposes_score_components() -> None:
    graph = make_graph(
        constraint(
            "feature_enabled => x % y == 0",
            ("feature_enabled", "x", "y"),
            kind=ConstraintKind.CONDITIONAL,
            confidence=0.5,
        )
    )

    candidate = select_interventions(
        graph,
        {"feature_enabled": False, "x": 4, "y": 2},
    ).candidates[0]

    components = dict(candidate.score_components)
    assert components["guard"] > 0
    assert components["interaction"] > 0
    assert components["uncertainty"] == 1.0
    assert candidate.intervention.satisfying.assignment["feature_enabled"] is True
    assert candidate.intervention.violating.assignment["feature_enabled"] is True


def test_selection_limit_is_deterministic() -> None:
    graph = make_graph(
        constraint("a % 2 == 0", ("a",)),
        constraint("b % 2 == 0", ("b",)),
        constraint("c % 2 == 0", ("c",)),
    )

    first = select_interventions(graph, {"a": 2, "b": 2, "c": 2}, limit=2)
    second = select_interventions(graph, {"a": 2, "b": 2, "c": 2}, limit=2)

    assert len(first.candidates) == 2
    assert [item.edge_id for item in first.candidates] == [
        item.edge_id for item in second.candidates
    ]
