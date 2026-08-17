from __future__ import annotations

import pytest

from configfuzz.dependencies import DependencyGraph, DependencyStatus
from configfuzz.graph_solver import SolveStatus, normalize_context, solve_graph_mutation
from configfuzz.model import Constraint, ConstraintKind, ConstraintSet


def make_graph(*constraints: Constraint) -> DependencyGraph:
    sets: dict[str, ConstraintSet] = {}
    for constraint in constraints:
        for parameter in constraint.parameters:
            result = sets.setdefault(parameter, ConstraintSet(parameter=parameter))
            result.add(constraint)
    return DependencyGraph.from_constraint_sets(sets.values())


def constraint(
    expression: str,
    kind: ConstraintKind,
    parameters: tuple[str, ...],
    *,
    confidence: float = 0.9,
) -> Constraint:
    return Constraint(
        expression=expression,
        kind=kind,
        parameters=parameters,
        confidence=confidence,
    )


def test_z3_joint_solver_preserves_divisibility_and_alignment() -> None:
    graph = make_graph(
        constraint("hidden_size: integer", ConstraintKind.TYPE, ("hidden_size",)),
        constraint(
            "tensor_model_parallel_size: integer",
            ConstraintKind.TYPE,
            ("tensor_model_parallel_size",),
        ),
        constraint(
            "hidden_size % tensor_model_parallel_size == 0",
            ConstraintKind.RELATION,
            ("hidden_size", "tensor_model_parallel_size"),
        ),
        constraint(
            "hidden_size % 8 == 0",
            ConstraintKind.RELATION,
            ("hidden_size",),
        ),
    )

    plan = solve_graph_mutation(
        graph,
        {"hidden_size": 4096, "tensor_model_parallel_size": 4},
        "tensor_model_parallel_size",
        6,
    )

    assert plan.status is SolveStatus.SAT
    assert plan.changes == {
        "tensor_model_parallel_size": 6,
        "hidden_size": 4104,
    }
    assert not plan.unsupported_edges


def test_z3_solver_respects_conditional_constraints() -> None:
    graph = make_graph(
        constraint("feature: boolean", ConstraintKind.TYPE, ("feature",)),
        constraint("parallel_size: integer", ConstraintKind.TYPE, ("parallel_size",)),
        constraint(
            "feature => parallel_size > 1",
            ConstraintKind.CONDITIONAL,
            ("feature", "parallel_size"),
        ),
    )

    plan = solve_graph_mutation(
        graph,
        {"feature": False, "parallel_size": 1},
        "feature",
        True,
    )

    assert plan.status is SolveStatus.SAT
    assert plan.changes == {"feature": True, "parallel_size": 2}


def test_z3_solver_reports_unsatisfiable_requested_value() -> None:
    graph = make_graph(
        constraint("x: integer", ConstraintKind.TYPE, ("x",)),
        constraint("x > 0", ConstraintKind.RANGE, ("x",)),
        constraint("x < 3", ConstraintKind.RANGE, ("x",)),
    )

    plan = solve_graph_mutation(
        graph,
        {"x": 1},
        "x",
        4,
        static_as_hard=True,
    )

    assert plan.status is SolveStatus.UNSAT
    assert plan.reason is not None


def test_static_candidates_are_soft_by_default() -> None:
    graph = make_graph(
        constraint("x: integer", ConstraintKind.TYPE, ("x",)),
        constraint("x > 0", ConstraintKind.RANGE, ("x",)),
        constraint("x < 3", ConstraintKind.RANGE, ("x",)),
    )

    plan = solve_graph_mutation(graph, {"x": 1}, "x", 4)

    assert plan.status is SolveStatus.SAT
    assert plan.changes == {"x": 4}
    assert len(plan.violated_soft_edges) == 1


def test_z3_solver_excludes_contradicted_edges() -> None:
    graph = make_graph(
        constraint("x: integer", ConstraintKind.TYPE, ("x",)),
        constraint("x > 10", ConstraintKind.RANGE, ("x",)),
    )
    range_edge = next(
        edge for edge in graph.edges.values() if edge.predicate == "x > 10"
    )
    graph.update_status(range_edge.id, DependencyStatus.CONTRADICTED)

    plan = solve_graph_mutation(graph, {"x": 12}, "x", 1)

    assert plan.status is SolveStatus.SAT
    assert plan.changes == {"x": 1}
    assert range_edge.id in plan.excluded_edges


def test_z3_solver_lists_unsupported_edges_without_approximating() -> None:
    graph = make_graph(
        constraint("pp: integer", ConstraintKind.TYPE, ("pp",)),
        constraint(
            "len(layout) % pp == 0",
            ConstraintKind.RELATION,
            ("layout", "pp"),
        ),
    )

    plan = solve_graph_mutation(graph, {"pp": 2, "layout": [1, 2]}, "pp", 4)

    assert plan.status is SolveStatus.SAT
    assert len(plan.unsupported_edges) == 1


def test_normalize_context_resolves_unique_nested_leaf_names() -> None:
    graph = make_graph(
        constraint("hidden_size: integer", ConstraintKind.TYPE, ("hidden_size",)),
        constraint(
            "model.hidden_size > 0",
            ConstraintKind.RELATION,
            ("model.hidden_size",),
        ),
    )

    context = normalize_context(graph, {"model": {"hidden_size": 4096}})

    assert context["hidden_size"] == 4096
    assert context["model.hidden_size"] == 4096


def test_missing_fixed_guard_context_is_not_solver_invented() -> None:
    graph = make_graph(
        constraint("feature: boolean", ConstraintKind.TYPE, ("feature",)),
        constraint("x: integer", ConstraintKind.TYPE, ("x",)),
        constraint(
            "feature => x > 1",
            ConstraintKind.CONDITIONAL,
            ("feature", "x"),
        ),
    )

    plan = solve_graph_mutation(graph, {"x": 1}, "x", 2)

    conditional = next(edge for edge in graph.edges.values() if edge.guard == "feature")
    assert conditional.id in plan.unsupported_edges
    assert "feature" in plan.missing_context


def test_z3_sort_mismatch_is_reported_as_unsupported() -> None:
    graph = make_graph(
        constraint("x: integer", ConstraintKind.TYPE, ("x",)),
        constraint("x in {'a'}", ConstraintKind.ENUM, ("x",)),
    )

    plan = solve_graph_mutation(graph, {"x": 1}, "x", 2)

    enum_edge = next(
        edge for edge in graph.edges.values() if edge.relation.value == "enum"
    )
    assert enum_edge.id in plan.unsupported_edges


def test_semantic_anchor_is_hard_preserved() -> None:
    graph = make_graph(
        constraint("hidden_size: integer", ConstraintKind.TYPE, ("hidden_size",)),
        constraint(
            "tensor_parallel_size: integer",
            ConstraintKind.TYPE,
            ("tensor_parallel_size",),
        ),
        constraint(
            "hidden_size % tensor_parallel_size == 0",
            ConstraintKind.RELATION,
            ("hidden_size", "tensor_parallel_size"),
        ),
    )

    plan = solve_graph_mutation(
        graph,
        {"hidden_size": 12, "tensor_parallel_size": 3},
        "tensor_parallel_size",
        5,
        semantic_anchors=("hidden_size",),
    )

    assert plan.status is SolveStatus.SAT
    assert plan.changes == {"tensor_parallel_size": 5}
    assert plan.semantic_anchors == ("hidden_size",)
    relation = next(
        edge
        for edge in graph.edges.values()
        if edge.predicate == "hidden_size % tensor_parallel_size == 0"
    )
    assert relation.id in plan.violated_soft_edges


def test_high_confidence_candidate_precedes_edit_locality() -> None:
    graph = make_graph(
        constraint("hidden_size: integer", ConstraintKind.TYPE, ("hidden_size",)),
        constraint(
            "tensor_parallel_size: integer",
            ConstraintKind.TYPE,
            ("tensor_parallel_size",),
        ),
        constraint(
            "hidden_size % tensor_parallel_size == 0",
            ConstraintKind.RELATION,
            ("hidden_size", "tensor_parallel_size"),
            confidence=0.95,
        ),
    )

    plan = solve_graph_mutation(
        graph,
        {"hidden_size": 12, "tensor_parallel_size": 3},
        "tensor_parallel_size",
        5,
    )

    assert plan.status is SolveStatus.SAT
    assert plan.changes == {"tensor_parallel_size": 5, "hidden_size": 10}
    relation = next(
        edge
        for edge in graph.edges.values()
        if edge.predicate == "hidden_size % tensor_parallel_size == 0"
    )
    assert relation.id in plan.high_confidence_soft_edges
    assert relation.id not in plan.violated_soft_edges


def test_local_solver_can_repair_driver_when_requested_target_is_dependent() -> None:
    graph = make_graph(
        constraint("hidden_size: integer", ConstraintKind.TYPE, ("hidden_size",)),
        constraint("num_heads: integer", ConstraintKind.TYPE, ("num_heads",)),
        constraint(
            "hidden_size % num_heads == 0",
            ConstraintKind.RELATION,
            ("hidden_size", "num_heads"),
            confidence=0.95,
        ),
    )

    plan = solve_graph_mutation(
        graph,
        {"hidden_size": 512, "num_heads": 8},
        "hidden_size",
        513,
    )

    assert plan.status is SolveStatus.SAT
    assert plan.changes["hidden_size"] == 513
    assert plan.changes["num_heads"] in {1, 3, 9, 19, 27, 57, 171, 513}


def test_global_solver_never_invents_mutations_for_unbound_graph_parameters() -> None:
    graph = make_graph(
        constraint("x: integer", ConstraintKind.TYPE, ("x",)),
        constraint("missing: integer", ConstraintKind.TYPE, ("missing",)),
        constraint("missing >= 0", ConstraintKind.RANGE, ("missing",)),
    )

    plan = solve_graph_mutation(
        graph,
        {"x": 1},
        "x",
        2,
        mutable_parameters=("x", "missing"),
    )

    assert plan.status is SolveStatus.SAT
    assert plan.changes == {"x": 2}
    assert "missing" not in plan.mutable_parameters


def test_local_solver_does_not_walk_unrelated_transitive_hyperedges() -> None:
    graph = make_graph(
        constraint("x: integer", ConstraintKind.TYPE, ("x",)),
        constraint("y: integer", ConstraintKind.TYPE, ("y",)),
        constraint("z: integer", ConstraintKind.TYPE, ("z",)),
        constraint("x == y", ConstraintKind.RELATION, ("x", "y")),
        constraint("y == z", ConstraintKind.RELATION, ("y", "z")),
    )

    plan = solve_graph_mutation(
        graph,
        {"x": 1, "y": 1, "z": 1},
        "x",
        2,
    )

    assert plan.status is SolveStatus.SAT
    assert "y" in plan.mutable_parameters
    assert "z" not in plan.mutable_parameters


def test_low_confidence_preference_follows_locality_objectives() -> None:
    graph = make_graph(
        constraint("hidden_size: integer", ConstraintKind.TYPE, ("hidden_size",)),
        constraint(
            "tensor_parallel_size: integer",
            ConstraintKind.TYPE,
            ("tensor_parallel_size",),
        ),
        constraint(
            "hidden_size % tensor_parallel_size == 0",
            ConstraintKind.RELATION,
            ("hidden_size", "tensor_parallel_size"),
            confidence=0.2,
        ),
    )

    plan = solve_graph_mutation(
        graph,
        {"hidden_size": 12, "tensor_parallel_size": 3},
        "tensor_parallel_size",
        5,
    )

    assert plan.status is SolveStatus.SAT
    assert plan.changes == {"tensor_parallel_size": 5}
    relation = next(
        edge
        for edge in graph.edges.values()
        if edge.predicate == "hidden_size % tensor_parallel_size == 0"
    )
    assert relation.id in plan.low_confidence_preference_edges
    assert relation.id in plan.violated_soft_edges


def test_mutation_solver_timeout_must_be_positive() -> None:
    graph = make_graph(constraint("x > 0", ConstraintKind.RANGE, ("x",)))

    with pytest.raises(ValueError, match="timeout"):
        solve_graph_mutation(graph, {"x": 1}, "x", 2, timeout_ms=0)
