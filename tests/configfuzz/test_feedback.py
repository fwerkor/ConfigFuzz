from __future__ import annotations

from configfuzz.dependencies import DependencyGraph, DependencyStatus
from configfuzz.feedback import apply_probe_feedback
from configfuzz.model import Constraint, ConstraintKind, ConstraintSet
from configfuzz.outcomes import ClassifiedOutcome, OutcomeLabel, ProcessObservation
from configfuzz.probing import ProbeSample


def graph_with_edges(*constraints: Constraint) -> DependencyGraph:
    sets: dict[str, ConstraintSet] = {}
    for constraint in constraints:
        for parameter in constraint.parameters:
            sets.setdefault(parameter, ConstraintSet(parameter=parameter)).add(
                constraint
            )
    return DependencyGraph.from_constraint_sets(sets.values())


def edge_constraint(expression: str, parameters: tuple[str, ...]) -> Constraint:
    return Constraint(
        expression=expression,
        kind=ConstraintKind.RELATION,
        parameters=parameters,
        confidence=0.7,
    )


def sample(
    parameter: str,
    value: object,
    label: OutcomeLabel,
    *,
    duration_seconds: float = 0.01,
    assignments: tuple[tuple[str, object], ...] = (),
    intervention_id: str | None = None,
    intervention_edge_id: str | None = None,
    intervention_role: str | None = None,
    provenance_matched: bool = False,
) -> ProbeSample:
    return ProbeSample(
        parameter=parameter,
        value=value,
        observation=ProcessObservation(
            argv=("probe",),
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=duration_seconds,
        ),
        outcome=ClassifiedOutcome(label=label, reason="test"),
        assignments=assignments,
        intervention_id=intervention_id,
        intervention_edge_id=intervention_edge_id,
        intervention_role=intervention_role,
        provenance_matched=provenance_matched,
    )


def test_consistency_and_isolated_violation_do_not_confirm_edge() -> None:
    graph = graph_with_edges(
        edge_constraint(
            "hidden_size % tensor_model_parallel_size == 0",
            ("hidden_size", "tensor_model_parallel_size"),
        )
    )

    report = apply_probe_feedback(
        graph,
        {"hidden_size": 16, "tensor_model_parallel_size": 4},
        [
            sample("hidden_size", 4, OutcomeLabel.VALID),
            sample("hidden_size", 8, OutcomeLabel.VALID),
            sample("hidden_size", 12, OutcomeLabel.VALID),
            sample("hidden_size", 5, OutcomeLabel.INVALID),
        ],
    )

    edge = next(iter(graph.edges.values()))
    assert edge.status is DependencyStatus.DYNAMICALLY_SUPPORTED
    assert edge.id in report.supported_edges
    assert edge.id not in report.confirmed_edges
    stats = graph.metadata["runtime_feedback"]["edges"][edge.id]
    assert stats["consistent_valid"] == 3
    assert stats["isolated_violation"] == 1
    assert stats["paired_intervention"] == 0


def test_provenance_matched_paired_intervention_confirms_edge() -> None:
    graph = graph_with_edges(
        edge_constraint(
            "hidden_size % tensor_model_parallel_size == 0",
            ("hidden_size", "tensor_model_parallel_size"),
        )
    )
    edge = next(iter(graph.edges.values()))

    report = apply_probe_feedback(
        graph,
        {"hidden_size": 16, "tensor_model_parallel_size": 4},
        [
            sample(
                "hidden_size",
                16,
                OutcomeLabel.VALID,
                intervention_id="pair-1",
                intervention_edge_id=edge.id,
                intervention_role="satisfying",
            ),
            sample(
                "hidden_size",
                18,
                OutcomeLabel.INVALID,
                intervention_id="pair-1",
                intervention_edge_id=edge.id,
                intervention_role="violating",
                provenance_matched=True,
            ),
        ],
    )

    updated = graph.edges[edge.id]
    assert updated.status is DependencyStatus.CONFIRMED
    assert edge.id in report.confirmed_edges
    assert report.paired_interventions == 1
    stats = graph.metadata["runtime_feedback"]["edges"][edge.id]
    assert stats["paired_intervention"] == 1
    assert stats["provenance_matched_rejection"] == 1


def test_paired_intervention_without_provenance_match_is_not_confirmed() -> None:
    graph = graph_with_edges(edge_constraint("x % 2 == 0", ("x",)))
    edge = next(iter(graph.edges.values()))

    report = apply_probe_feedback(
        graph,
        {"x": 2},
        [
            sample(
                "x",
                2,
                OutcomeLabel.VALID,
                intervention_id="pair-2",
                intervention_edge_id=edge.id,
                intervention_role="satisfying",
            ),
            sample(
                "x",
                3,
                OutcomeLabel.INVALID,
                intervention_id="pair-2",
                intervention_edge_id=edge.id,
                intervention_role="violating",
            ),
        ],
    )

    assert graph.edges[edge.id].status is DependencyStatus.DYNAMICALLY_SUPPORTED
    assert not report.confirmed_edges
    assert report.paired_interventions == 0


def test_valid_counterexample_marks_scope_disputed_not_globally_contradicted() -> None:
    graph = graph_with_edges(
        edge_constraint(
            "hidden_size % tensor_model_parallel_size == 0",
            ("hidden_size", "tensor_model_parallel_size"),
        )
    )

    report = apply_probe_feedback(
        graph,
        {"hidden_size": 16, "tensor_model_parallel_size": 4},
        [sample("hidden_size", 5, OutcomeLabel.VALID)],
    )

    edge = next(iter(graph.edges.values()))
    assert edge.status is DependencyStatus.SCOPE_DISPUTED
    assert edge.id in report.scope_disputed_edges
    assert edge.id not in report.contradicted_edges
    assert edge.confidence < 0.7


def test_invalid_sample_with_multiple_violations_is_ambiguous() -> None:
    graph = graph_with_edges(
        edge_constraint("x % 2 == 0", ("x",)),
        edge_constraint("x > 0", ("x",)),
    )

    report = apply_probe_feedback(
        graph,
        {"x": 2},
        [sample("x", -1, OutcomeLabel.INVALID)],
    )

    assert report.ambiguous_invalid_samples == 1
    assert all(
        edge.status is DependencyStatus.STATIC_CANDIDATE
        for edge in graph.edges.values()
    )
    assert all(
        graph.metadata["runtime_feedback"]["edges"][edge.id]["isolated_violation"] == 0
        for edge in graph.edges.values()
    )


def test_potential_bug_does_not_become_invalid_domain_evidence() -> None:
    graph = graph_with_edges(edge_constraint("x % 2 == 0", ("x",)))

    report = apply_probe_feedback(
        graph,
        {"x": 2},
        [sample("x", 4, OutcomeLabel.POTENTIAL_BUG)],
    )

    edge = next(iter(graph.edges.values()))
    assert edge.status is DependencyStatus.STATIC_CANDIDATE
    assert not report.supported_edges
    stats = graph.metadata["runtime_feedback"]["edges"][edge.id]
    assert stats["bug_preserving"] == 1


def test_unknown_samples_are_ignored() -> None:
    graph = graph_with_edges(edge_constraint("x > 0", ("x",)))

    report = apply_probe_feedback(
        graph,
        {"x": 1},
        [sample("x", -1, OutcomeLabel.UNKNOWN)],
    )

    assert report.ignored_samples == 1
    assert "runtime_feedback" in graph.metadata
    assert graph.metadata["runtime_feedback"]["outcomes"] == {"unknown": 1}


def test_feedback_batch_is_not_counted_twice() -> None:
    graph = graph_with_edges(edge_constraint("x > 0", ("x",)))
    first_samples = [sample("x", 1, OutcomeLabel.VALID, duration_seconds=0.01)]
    repeated_samples = [sample("x", 1, OutcomeLabel.VALID, duration_seconds=9.0)]

    first = apply_probe_feedback(graph, {"x": 1}, first_samples)
    second = apply_probe_feedback(graph, {"x": 1}, repeated_samples)

    assert not first.duplicate_batch
    assert second.duplicate_batch
    assert first.batch_id == second.batch_id
    assert graph.metadata["runtime_feedback"]["runs"] == 1
    edge = next(iter(graph.edges.values()))
    assert graph.metadata["runtime_feedback"]["edges"][edge.id]["evaluated"] == 1


def test_feedback_ignores_out_of_stage_relation() -> None:
    graph = graph_with_edges(edge_constraint("x > 0", ("x",)))
    edge = next(iter(graph.edges.values()))
    graph.edges[edge.id] = edge.__class__(
        id=edge.id,
        expression=edge.expression,
        predicate=edge.predicate,
        relation=edge.relation,
        participants=edge.participants,
        drivers=edge.drivers,
        dependents=edge.dependents,
        guard=edge.guard,
        status=edge.status,
        confidence=edge.confidence,
        scope=(("execution_stage", "inference"),),
        components=edge.components,
        evidence=edge.evidence,
    )

    report = apply_probe_feedback(
        graph,
        {"x": 2, "execution_stage": "training"},
        [sample("x", -1, OutcomeLabel.VALID)],
    )

    assert report.evaluated_samples == 0
    assert report.ignored_samples == 1
    assert graph.edges[edge.id].status is DependencyStatus.STATIC_CANDIDATE
