from __future__ import annotations

import hashlib
import json
from collections import Counter
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
    contradicted_edges: tuple[str, ...]
    ambiguous_invalid_samples: int
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
            "contradicted_edges": list(self.contradicted_edges),
            "ambiguous_invalid_samples": self.ambiguous_invalid_samples,
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
            contradicted_edges=(),
            ambiguous_invalid_samples=0,
            status_changes=(),
        )
    evaluated_samples = 0
    ignored_samples = 0
    ambiguous_invalid_samples = 0

    for sample in samples:
        if sample.outcome.label is OutcomeLabel.UNKNOWN:
            ignored_samples += 1
            continue
        edges = graph.constraints_for(sample.parameter)
        if not edges:
            ignored_samples += 1
            continue
        context = dict(base_context)
        context[sample.parameter] = sample.value
        evaluations = [
            (edge, graph.evaluate_edge(edge, context))
            for edge in edges
        ]
        decisive = [
            (edge, evaluation)
            for edge, evaluation in evaluations
            if evaluation.active is True and evaluation.satisfied is not None
        ]
        if not decisive:
            ignored_samples += 1
            continue
        evaluated_samples += 1

        if sample.outcome.label is OutcomeLabel.VALID:
            for edge, evaluation in decisive:
                stats = _stats_for(edge_stats, edge.id, edge.confidence)
                stats["evaluated"] += 1
                if evaluation.satisfied is True:
                    stats["valid_support"] += 1
                else:
                    stats["valid_contradiction"] += 1
            continue

        if sample.outcome.label is OutcomeLabel.INVALID:
            false_edges = [edge for edge, evaluation in decisive if evaluation.satisfied is False]
            for edge, _ in decisive:
                _stats_for(edge_stats, edge.id, edge.confidence)["evaluated"] += 1
            if len(false_edges) == 1:
                _stats_for(edge_stats, false_edges[0].id, false_edges[0].confidence)[
                    "isolated_invalid_support"
                ] += 1
            elif len(false_edges) > 1:
                ambiguous_invalid_samples += 1
                for edge in false_edges:
                    _stats_for(edge_stats, edge.id, edge.confidence)[
                        "ambiguous_invalid"
                    ] += 1
            continue

        if sample.outcome.label is OutcomeLabel.POTENTIAL_BUG:
            if all(evaluation.satisfied is True for _, evaluation in decisive):
                for edge, _ in decisive:
                    stats = _stats_for(edge_stats, edge.id, edge.confidence)
                    stats["evaluated"] += 1
                    stats["bug_preserving"] += 1
            continue

    status_changes: list[tuple[str, str, str]] = []
    supported: list[str] = []
    confirmed: list[str] = []
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
                item.kind is EvidenceKind.DYNAMIC
                and item.source == "runtime-feedback"
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
        contradicted_edges=tuple(sorted(contradicted)),
        ambiguous_invalid_samples=ambiguous_invalid_samples,
        status_changes=tuple(sorted(status_changes)),
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
        "valid_support": 0,
        "valid_contradiction": 0,
        "isolated_invalid_support": 0,
        "ambiguous_invalid": 0,
        "bug_preserving": 0,
    }
    edge_stats[edge_id] = stats
    return stats


def _status_from_stats(
    current: DependencyStatus,
    stats: Mapping[str, Any],
) -> DependencyStatus:
    contradictions = int(stats.get("valid_contradiction", 0))
    valid_support = int(stats.get("valid_support", 0))
    invalid_support = int(stats.get("isolated_invalid_support", 0))
    if contradictions > 0:
        return DependencyStatus.CONTRADICTED
    if current is DependencyStatus.ENVIRONMENT_SPECIFIC:
        return current
    if valid_support >= 3 and invalid_support >= 1:
        return DependencyStatus.CONFIRMED
    if valid_support >= 2 or invalid_support >= 1:
        return DependencyStatus.DYNAMICALLY_SUPPORTED
    return current


def _confidence_from_stats(stats: Mapping[str, Any]) -> float:
    confidence = float(stats.get("base_confidence", 0.5))
    confidence += min(0.12, 0.02 * int(stats.get("valid_support", 0)))
    confidence += min(0.24, 0.08 * int(stats.get("isolated_invalid_support", 0)))
    confidence -= min(0.9, 0.35 * int(stats.get("valid_contradiction", 0)))
    return round(max(0.05, min(0.99, confidence)), 4)


def _feedback_detail(stats: Mapping[str, Any]) -> str:
    return (
        f"valid_support={int(stats.get('valid_support', 0))}; "
        f"isolated_invalid_support={int(stats.get('isolated_invalid_support', 0))}; "
        f"valid_contradiction={int(stats.get('valid_contradiction', 0))}; "
        f"ambiguous_invalid={int(stats.get('ambiguous_invalid', 0))}; "
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
