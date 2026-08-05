from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from configfuzz.dependencies import DependencyGraph
from configfuzz.feedback import FeedbackReport, apply_probe_feedback
from configfuzz.intervention_runner import (
    InterventionExecutionManifest,
    run_intervention,
)
from configfuzz.probing import ProbeSample
from configfuzz.selection import InterventionCandidate, select_interventions


@dataclass(frozen=True, slots=True)
class ActiveValidationRound:
    index: int
    candidate: InterventionCandidate
    samples: tuple[ProbeSample, ...]
    feedback: FeedbackReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "candidate": self.candidate.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
            "feedback": self.feedback.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ActiveValidationResult:
    stop_reason: str
    rounds: tuple[ActiveValidationRound, ...]
    attempted_edges: tuple[str, ...]
    graph: DependencyGraph
    round_budget: int
    solver_timeout_ms: int

    def to_dict(self) -> dict[str, Any]:
        status_counts = Counter(edge.status.value for edge in self.graph.edges.values())
        return {
            "summary": {
                "stop_reason": self.stop_reason,
                "rounds_executed": len(self.rounds),
                "round_budget": self.round_budget,
                "solver_timeout_ms": self.solver_timeout_ms,
                "attempted_edges": list(self.attempted_edges),
                "edge_statuses": dict(sorted(status_counts.items())),
            },
            "rounds": [round_.to_dict() for round_ in self.rounds],
            "dependency_graph": self.graph.to_dict(),
        }


def run_active_validation(
    graph: DependencyGraph,
    manifest: InterventionExecutionManifest,
    *,
    max_rounds: int = 10,
    solver_timeout_ms: int = 1000,
) -> ActiveValidationResult:
    if max_rounds <= 0:
        raise ValueError("active-validation round budget must be positive")
    if solver_timeout_ms <= 0:
        raise ValueError("intervention solver timeout must be positive")
    baseline = _load_baseline(manifest)
    attempted: set[str] = set()
    rounds: list[ActiveValidationRound] = []
    stop_reason = "budget_exhausted"

    for index in range(1, max_rounds + 1):
        queue = select_interventions(
            graph,
            baseline,
            limit=1,
            excluded_edge_ids=attempted,
            solver_timeout_ms=solver_timeout_ms,
        )
        if not queue.candidates:
            stop_reason = "no_executable_candidates"
            break
        candidate = queue.candidates[0]
        attempted.add(candidate.edge_id)
        samples = tuple(
            run_intervention(
                {"intervention": candidate.intervention.to_dict()},
                manifest,
            )
        )
        feedback = apply_probe_feedback(graph, baseline, samples)
        rounds.append(
            ActiveValidationRound(
                index=index,
                candidate=candidate,
                samples=samples,
                feedback=feedback,
            )
        )

    return ActiveValidationResult(
        stop_reason=stop_reason,
        rounds=tuple(rounds),
        attempted_edges=tuple(round_.candidate.edge_id for round_ in rounds),
        graph=graph,
        round_budget=max_rounds,
        solver_timeout_ms=solver_timeout_ms,
    )


def _load_baseline(manifest: InterventionExecutionManifest) -> Mapping[str, Any]:
    payload = json.loads(manifest.baseline_config.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("active-validation baseline must be a JSON object")
    return payload
