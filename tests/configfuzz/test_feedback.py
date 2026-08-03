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
            sets.setdefault(parameter, ConstraintSet(parameter=parameter)).add(constraint)
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
    )


def test_feedback_confirms_edge_from_valid_and_isolated_invalid_samples() -> None:
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
    assert edge.status is DependencyStatus.CONFIRMED
    assert edge.id in report.confirmed_edges
    assert edge.confidence > 0.7
    assert graph.metadata["runtime_feedback"]["edges"][edge.id][
        "isolated_invalid_support"
    ] == 1


def test_valid_sample_that_violates_edge_marks_it_contradicted() -> None:
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
    assert edge.status is DependencyStatus.CONTRADICTED
    assert edge.id in report.contradicted_edges
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
        graph.metadata["runtime_feedback"]["edges"][edge.id][
            "isolated_invalid_support"
        ]
        == 0
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
