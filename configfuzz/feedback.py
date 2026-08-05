from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from configfuzz.dependencies import DependencyGraph, DependencyStatus
from configfuzz.graph_solver import normalize_context
from configfuzz.model import Evidence, EvidenceKind
from configfuzz.outcomes import OutcomeLabel
from configfuzz.probing import ProbeSample


@dataclass(frozen=True, slots=True)
class FeedbackReport:
    batch_id: str
    duplicate_batch: bool
    samples: int
    evaluated_samples: int
    ignored_samples: int
    supported_edges: tuple[str, ...]
    confirmed_edges: tuple[str, ...]
    scope_disputed_edges: tuple[str, ...]
    contradicted_edges: tuple[str, ...]
    ambiguous_invalid_samples: int
    paired_interventions: int
    status_changes: tuple[tuple[str, str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "duplicate_batch": self.duplicate_batch,
            "samples": self.samples,
            "evaluated_samples": self.evaluated_samples,
            "ignored_samples": self.ignored_samples,
            "supported_edges": list(self.supported_edges),
            "confirmed_edges": list(self.confirmed_edges),
            "scope_disputed_edges": list(self.scope_disputed_edges),
            "contradicted_edges": list(self.contradicted_edges),
            "ambiguous_invalid_samples": self.ambiguous_invalid_samples,
            "paired_interventions": self.paired_interventions,
            "status_changes": [
                {"edge_id": edge_id, "from": old, "to": new}
                for edge_id, old, new in self.status_changes
            ],
        }


def apply_probe_feedback(
    graph: DependencyGraph,
    baseline: Mapping[str, Any],
    samples: Iterable[ProbeSample],
) -> FeedbackReport:
    """Apply execution evidence without conflating consistency with necessity.

    A valid sample satisfying an edge is only consistency evidence. An invalid
    sample that violates one *known* edge is only isolated-violation evidence,
    because an unrecovered relation may still explain the failure. Confirmation
    requires a paired intervention whose satisfying member is valid, whose
    violating member is rejected, and whose rejection matches the candidate's
    provenance. A valid counterexample disputes the current guard or scope
    instead of globally deleting the relation.
    """

    samples = list(samples)
    feedback = graph.metadata.setdefault("runtime_feedback", {})
    if not isinstance(feedback, dict):
        feedback = {}
        graph.metadata["runtime_feedback"] = feedback
    edge_stats = feedback.setdefault("edges", {})
    if not isinstance(edge_stats, dict):
        edge_stats = {}
        feedback["edges"] = edge_stats

    base_context = normalize_context(graph, baseline)
    batch_id = _feedback_batch_id(base_context, samples)
    applied_batches = feedback.setdefault("applied_batches", [])
    if not isinstance(applied_batches, list):
        applied_batches = []
        feedback["applied_batches"] = applied_batches
    if batch_id in applied_batches:
        return FeedbackReport(
            batch_id=batch_id,
            duplicate_batch=True,
            samples=len(samples),
            evaluated_samples=0,
            ignored_samples=len(samples),
            supported_edges=(),
            confirmed_edges=(),
            scope_disputed_edges=(),
            contradicted_edges=(),
            ambiguous_invalid_samples=0,
            paired_interventions=0,
            status_changes=(),
        )

    evaluated_samples = 0
    ignored_samples = 0
    ambiguous_invalid_samples = 0

    for sample in samples:
        if sample.outcome.label in {
            OutcomeLabel.UNKNOWN,
            OutcomeLabel.RESOURCE_FAILURE,
            OutcomeLabel.INFRASTRUCTURE_FAILURE,
        }:
            ignored_samples += 1
            continue
        decisive = _decisive_evaluations(graph, base_context, sample)
        if not decisive:
            ignored_samples += 1
            continue
        evaluated_samples += 1

        if sample.outcome.label is OutcomeLabel.VALID:
            for edge, evaluation in decisive:
                stats = _stats_for(edge_stats, edge.id, edge.confidence)
                stats["evaluated"] += 1
                if evaluation.satisfied is True:
                    stats["consistent_valid"] += 1
                else:
                    stats["valid_counterexample"] += 1
            continue

        if sample.outcome.label is OutcomeLabel.INVALID:
            false_edges = [
                edge for edge, evaluation in decisive if evaluation.satisfied is False
            ]
            for edge, _ in decisive:
                _stats_for(edge_stats, edge.id, edge.confidence)["evaluated"] += 1
            if len(false_edges) == 1:
                _stats_for(edge_stats, false_edges[0].id, false_edges[0].confidence)[
                    "isolated_violation"
                ] += 1
            elif len(false_edges) > 1:
                ambiguous_invalid_samples += 1
                for edge in false_edges:
                    _stats_for(edge_stats, edge.id, edge.confidence)[
                        "ambiguous_invalid"
                    ] += 1
            continue

        if sample.outcome.label in {
            OutcomeLabel.UNEXPLAINED_FAILURE,
            OutcomeLabel.POTENTIAL_BUG,
        }:
            if all(evaluation.satisfied is True for _, evaluation in decisive):
                for edge, _ in decisive:
                    stats = _stats_for(edge_stats, edge.id, edge.confidence)
                    stats["evaluated"] += 1
                    key = (
                        "bug_preserving"
                        if sample.outcome.label is OutcomeLabel.POTENTIAL_BUG
                        else "unexplained_preserving"
                    )
                    stats[key] += 1

    paired_interventions = _apply_paired_interventions(
        graph,
        base_context,
        samples,
        edge_stats,
    )

    status_changes: list[tuple[str, str, str]] = []
    supported: list[str] = []
    confirmed: list[str] = []
    scope_disputed: list[str] = []
    contradicted: list[str] = []
    for edge_id, edge in list(graph.edges.items()):
        raw_stats = edge_stats.get(edge_id)
        if not isinstance(raw_stats, dict):
            continue
        old_status = edge.status
        new_status = _status_from_stats(edge.status, raw_stats)
        confidence = _confidence_from_stats(raw_stats)
        evidence = tuple(
            item
            for item in edge.evidence
            if not (
                item.kind is EvidenceKind.DYNAMIC and item.source == "runtime-feedback"
            )
        )
        evidence = (
            *evidence,
            Evidence(
                kind=EvidenceKind.DYNAMIC,
                source="runtime-feedback",
                detail=_feedback_detail(raw_stats),
            ),
        )
        graph.edges[edge_id] = replace(
            edge,
            status=new_status,
            confidence=confidence,
            evidence=evidence,
        )
        if new_status is not old_status:
            status_changes.append((edge_id, old_status.value, new_status.value))
        if new_status is DependencyStatus.DYNAMICALLY_SUPPORTED:
            supported.append(edge_id)
        elif new_status is DependencyStatus.CONFIRMED:
            confirmed.append(edge_id)
        elif new_status is DependencyStatus.SCOPE_DISPUTED:
            scope_disputed.append(edge_id)
        elif new_status is DependencyStatus.CONTRADICTED:
            contradicted.append(edge_id)

    feedback["runs"] = int(feedback.get("runs", 0)) + 1
    feedback["samples"] = int(feedback.get("samples", 0)) + len(samples)
    feedback["outcomes"] = _updated_outcome_counts(feedback.get("outcomes"), samples)
    applied_batches.append(batch_id)
    return FeedbackReport(
        batch_id=batch_id,
        duplicate_batch=False,
        samples=len(samples),
        evaluated_samples=evaluated_samples,
        ignored_samples=ignored_samples,
        supported_edges=tuple(sorted(supported)),
        confirmed_edges=tuple(sorted(confirmed)),
        scope_disputed_edges=tuple(sorted(scope_disputed)),
        contradicted_edges=tuple(sorted(contradicted)),
        ambiguous_invalid_samples=ambiguous_invalid_samples,
        paired_interventions=paired_interventions,
        status_changes=tuple(sorted(status_changes)),
    )


def _decisive_evaluations(
    graph: DependencyGraph,
    base_context: Mapping[str, Any],
    sample: ProbeSample,
    *,
    relevant_only: bool = True,
):
    context = dict(base_context)
    context.update(sample.configuration_updates)
    if relevant_only:
        edge_map = {
            edge.id: edge
            for name in sample.configuration_updates
            for edge in graph.constraints_for(name)
        }
        edges = edge_map.values()
    else:
        edges = graph.edges.values()
    return [
        (edge, evaluation)
        for edge in edges
        if (
            (evaluation := graph.evaluate_edge(edge, context)).active is True
            and evaluation.satisfied is not None
        )
    ]


def _apply_paired_interventions(
    graph: DependencyGraph,
    base_context: Mapping[str, Any],
    samples: Iterable[ProbeSample],
    edge_stats: dict[str, Any],
) -> int:
    groups: dict[tuple[str, str], list[ProbeSample]] = defaultdict(list)
    for sample in samples:
        if sample.intervention_id and sample.intervention_edge_id:
            groups[(sample.intervention_id, sample.intervention_edge_id)].append(sample)

    accepted = 0
    for (_, edge_id), group in groups.items():
        edge = graph.edges.get(edge_id)
        if edge is None:
            continue
        positives = [
            item
            for item in group
            if (item.intervention_role or "").lower()
            in {"satisfying", "positive", "plus"}
        ]
        negatives = [
            item
            for item in group
            if (item.intervention_role or "").lower()
            in {"violating", "negative", "minus"}
        ]
        repaired = [
            item
            for item in group
            if (item.intervention_role or "").lower() in {"repaired", "repair"}
        ]
        matched = False
        for positive in positives:
            if positive.outcome.label is not OutcomeLabel.VALID:
                continue
            positive_eval = _evaluation_map(graph, base_context, positive)
            target_positive = positive_eval.get(edge_id)
            if target_positive is None or target_positive.satisfied is not True:
                continue
            if not _all_other_hard_edges_satisfied(graph, positive_eval, edge_id):
                continue
            for negative in negatives:
                if negative.outcome.label is not OutcomeLabel.INVALID:
                    continue
                if not negative.provenance_matched:
                    continue
                negative_eval = _evaluation_map(graph, base_context, negative)
                target_negative = negative_eval.get(edge_id)
                if target_negative is None or target_negative.satisfied is not False:
                    continue
                if not _all_other_hard_edges_satisfied(graph, negative_eval, edge_id):
                    continue
                if repaired and not any(
                    repair.outcome.label is OutcomeLabel.VALID
                    and (
                        repair_target := _evaluation_map(
                            graph, base_context, repair
                        ).get(edge_id)
                    )
                    is not None
                    and repair_target.satisfied is True
                    for repair in repaired
                ):
                    continue
                stats = _stats_for(edge_stats, edge_id, edge.confidence)
                stats["paired_intervention"] += 1
                stats["provenance_matched_rejection"] += 1
                matched = True
                accepted += 1
                break
            if matched:
                break
    return accepted


def _evaluation_map(
    graph: DependencyGraph,
    base_context: Mapping[str, Any],
    sample: ProbeSample,
):
    return {
        edge.id: evaluation
        for edge, evaluation in _decisive_evaluations(
            graph,
            base_context,
            sample,
            relevant_only=False,
        )
    }


def _all_other_hard_edges_satisfied(
    graph: DependencyGraph,
    evaluations: Mapping[str, Any],
    edge_id: str,
) -> bool:
    hard_statuses = {
        DependencyStatus.CONFIRMED,
        DependencyStatus.ENVIRONMENT_SPECIFIC,
    }
    return all(
        evaluation.satisfied is True
        for current_id, evaluation in evaluations.items()
        if current_id != edge_id and graph.edges[current_id].status in hard_statuses
    )


def _stats_for(
    edge_stats: dict[str, Any],
    edge_id: str,
    confidence: float,
) -> dict[str, Any]:
    existing = edge_stats.get(edge_id)
    if isinstance(existing, dict):
        return existing
    stats = {
        "base_confidence": confidence,
        "evaluated": 0,
        "consistent_valid": 0,
        "valid_counterexample": 0,
        "isolated_violation": 0,
        "paired_intervention": 0,
        "provenance_matched_rejection": 0,
        "ambiguous_invalid": 0,
        "unexplained_preserving": 0,
        "bug_preserving": 0,
    }
    edge_stats[edge_id] = stats
    return stats


def _status_from_stats(
    current: DependencyStatus,
    stats: Mapping[str, Any],
) -> DependencyStatus:
    counterexamples = int(stats.get("valid_counterexample", 0))
    paired = int(stats.get("paired_intervention", 0))
    provenance_matched = int(stats.get("provenance_matched_rejection", 0))
    consistent = int(stats.get("consistent_valid", 0))
    isolated = int(stats.get("isolated_violation", 0))
    if counterexamples > 0:
        return DependencyStatus.SCOPE_DISPUTED
    if paired > 0 and provenance_matched > 0:
        return DependencyStatus.CONFIRMED
    if current is DependencyStatus.ENVIRONMENT_SPECIFIC:
        return current
    if consistent >= 2 or isolated >= 1:
        return DependencyStatus.DYNAMICALLY_SUPPORTED
    return current


def _confidence_from_stats(stats: Mapping[str, Any]) -> float:
    confidence = float(stats.get("base_confidence", 0.5))
    confidence += min(0.08, 0.01 * int(stats.get("consistent_valid", 0)))
    confidence += min(0.09, 0.03 * int(stats.get("isolated_violation", 0)))
    confidence += min(0.35, 0.25 * int(stats.get("paired_intervention", 0)))
    confidence += min(
        0.2,
        0.15 * int(stats.get("provenance_matched_rejection", 0)),
    )
    confidence -= min(0.8, 0.35 * int(stats.get("valid_counterexample", 0)))
    return round(max(0.05, min(0.99, confidence)), 4)


def _feedback_detail(stats: Mapping[str, Any]) -> str:
    return (
        f"consistent_valid={int(stats.get('consistent_valid', 0))}; "
        f"isolated_violation={int(stats.get('isolated_violation', 0))}; "
        f"paired_intervention={int(stats.get('paired_intervention', 0))}; "
        "provenance_matched_rejection="
        f"{int(stats.get('provenance_matched_rejection', 0))}; "
        f"valid_counterexample={int(stats.get('valid_counterexample', 0))}; "
        f"ambiguous_invalid={int(stats.get('ambiguous_invalid', 0))}; "
        f"unexplained_preserving={int(stats.get('unexplained_preserving', 0))}; "
        f"bug_preserving={int(stats.get('bug_preserving', 0))}"
    )


def _updated_outcome_counts(
    existing: Any,
    samples: Iterable[ProbeSample],
) -> dict[str, int]:
    counts = Counter(existing if isinstance(existing, Mapping) else {})
    counts.update(sample.outcome.label.value for sample in samples)
    return dict(sorted((str(key), int(value)) for key, value in counts.items()))


def _feedback_batch_id(
    baseline: Mapping[str, Any],
    samples: Iterable[ProbeSample],
) -> str:
    payload = {
        "baseline": dict(sorted((str(key), value) for key, value in baseline.items())),
        "samples": [
            {
                "parameter": sample.parameter,
                "value": sample.value,
                "assignments": dict(sample.assignments),
                "intervention_id": sample.intervention_id,
                "intervention_edge_id": sample.intervention_edge_id,
                "intervention_role": sample.intervention_role,
                "provenance_matched": sample.provenance_matched,
                "outcome": sample.outcome.to_dict(),
                "observation": {
                    "argv": list(sample.observation.argv),
                    "returncode": sample.observation.returncode,
                    "stdout": sample.observation.stdout,
                    "stderr": sample.observation.stderr,
                    "timed_out": sample.observation.timed_out,
                },
            }
            for sample in samples
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]
