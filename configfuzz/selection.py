from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Collection
from typing import Any, Mapping

from configfuzz.dependencies import (
    DependencyEdge,
    DependencyGraph,
    DependencyRelation,
    DependencyStatus,
)
from configfuzz.graph_solver import (
    InterventionPlan,
    SolveStatus,
    design_edge_intervention,
)


_ELIGIBLE_STATUSES = {
    DependencyStatus.STATIC_CANDIDATE,
    DependencyStatus.DYNAMICALLY_SUPPORTED,
    DependencyStatus.ENVIRONMENT_SPECIFIC,
    DependencyStatus.SCOPE_DISPUTED,
}

_STATUS_PRIORITY = {
    DependencyStatus.SCOPE_DISPUTED: 1.1,
    DependencyStatus.STATIC_CANDIDATE: 1.0,
    DependencyStatus.DYNAMICALLY_SUPPORTED: 0.75,
    DependencyStatus.ENVIRONMENT_SPECIFIC: 0.6,
}

_RELATION_PRIORITY = {
    DependencyRelation.CONDITIONAL: 1.0,
    DependencyRelation.REQUIRES: 1.0,
    DependencyRelation.CONFLICTS: 1.0,
    DependencyRelation.PRODUCT_LIMIT: 0.95,
    DependencyRelation.RESOURCE: 0.95,
    DependencyRelation.ENVIRONMENT: 0.9,
    DependencyRelation.DIVISIBILITY: 0.9,
    DependencyRelation.ALIGNMENT: 0.8,
    DependencyRelation.EQUALITY: 0.75,
    DependencyRelation.BOUND: 0.65,
    DependencyRelation.RANGE: 0.55,
    DependencyRelation.ENUM: 0.45,
    DependencyRelation.TYPE: 0.2,
    DependencyRelation.OTHER: 0.35,
}


@dataclass(frozen=True, slots=True)
class InterventionCandidate:
    edge_id: str
    expression: str
    status: DependencyStatus
    relation: DependencyRelation
    confidence: float
    score: float
    score_components: tuple[tuple[str, float], ...]
    intervention: InterventionPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "expression": self.expression,
            "status": self.status.value,
            "relation": self.relation.value,
            "confidence": self.confidence,
            "score": self.score,
            "score_components": dict(self.score_components),
            "intervention": self.intervention.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class InterventionQueue:
    candidates: tuple[InterventionCandidate, ...]
    considered_edges: int
    skipped_status: int
    skipped_excluded: int
    skipped_infeasible: int
    skipped_uncompetitive: int
    solver_timeout_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "considered_edges": self.considered_edges,
                "selected_candidates": len(self.candidates),
                "skipped_status": self.skipped_status,
                "skipped_excluded": self.skipped_excluded,
                "skipped_infeasible": self.skipped_infeasible,
                "skipped_uncompetitive": self.skipped_uncompetitive,
                "solver_timeout_ms": self.solver_timeout_ms,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def select_interventions(
    graph: DependencyGraph,
    baseline: Mapping[str, Any],
    *,
    limit: int = 10,
    excluded_edge_ids: Collection[str] = (),
    solver_timeout_ms: int = 1000,
) -> InterventionQueue:
    if limit <= 0:
        raise ValueError("intervention selection limit must be positive")
    if solver_timeout_ms <= 0:
        raise ValueError("intervention solver timeout must be positive")
    candidates: list[InterventionCandidate] = []
    excluded = set(excluded_edge_ids)
    skipped_status = 0
    skipped_excluded = 0
    skipped_infeasible = 0
    eligible: list[tuple[float, DependencyEdge, dict[str, float]]] = []
    for edge in graph.edges.values():
        if edge.id in excluded:
            skipped_excluded += 1
            continue
        if edge.status not in _ELIGIBLE_STATUSES:
            skipped_status += 1
            continue
        static_components = _static_score_components(graph, edge)
        eligible.append((sum(static_components.values()), edge, static_components))

    eligible.sort(key=lambda item: (-item[0], item[1].id))
    skipped_uncompetitive = 0
    for index, (upper_bound, edge, static_components) in enumerate(eligible):
        if len(candidates) >= limit:
            threshold = sorted(
                candidates,
                key=lambda item: (-item.score, item.edge_id),
            )[limit - 1].score
            if upper_bound < threshold:
                skipped_uncompetitive = len(eligible) - index
                break
        plan = design_edge_intervention(
            graph,
            baseline,
            edge.id,
            timeout_ms=solver_timeout_ms,
        )
        if (
            plan.satisfying.status is not SolveStatus.SAT
            or plan.violating.status is not SolveStatus.SAT
        ):
            skipped_infeasible += 1
            continue
        components = {
            **static_components,
            **_plan_score_components(graph, plan),
        }
        score = round(sum(components.values()), 6)
        candidates.append(
            InterventionCandidate(
                edge_id=edge.id,
                expression=edge.expression,
                status=edge.status,
                relation=edge.relation,
                confidence=edge.confidence,
                score=score,
                score_components=tuple(sorted(components.items())),
                intervention=plan,
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.edge_id))
    return InterventionQueue(
        candidates=tuple(candidates[:limit]),
        considered_edges=len(graph.edges),
        skipped_status=skipped_status,
        skipped_excluded=skipped_excluded,
        skipped_infeasible=skipped_infeasible,
        skipped_uncompetitive=skipped_uncompetitive,
        solver_timeout_ms=solver_timeout_ms,
    )


def _static_score_components(
    graph: DependencyGraph,
    edge: DependencyEdge,
) -> dict[str, float]:
    uncertainty = max(0.0, 1.0 - abs(edge.confidence - 0.5) * 2.0)
    interaction = min(max(len(edge.participants) - 1, 0), 4) / 4.0
    centrality = min(len({
        neighbor
        for participant in edge.participants
        for neighbor in graph.related_parameters(participant)
    }), 8) / 8.0
    return {
        "status": 3.0 * _STATUS_PRIORITY[edge.status],
        "relation": 1.2 * _RELATION_PRIORITY[edge.relation],
        "uncertainty": uncertainty,
        "interaction": 0.8 * interaction,
        "guard": 0.5 if edge.guard is not None else 0.0,
        "centrality": 0.4 * centrality,
        "provenance": 0.2 if edge.evidence else 0.0,
    }


def _plan_score_components(
    graph: DependencyGraph,
    plan: InterventionPlan,
) -> dict[str, float]:
    changed = {
        name
        for case in (plan.satisfying, plan.violating)
        for name in case.changed_values
    }
    pair_cost = len(changed) / max(1, len(plan.mutable_parameters))
    unsupported = {
        edge_id
        for case in (plan.satisfying, plan.violating)
        for edge_id in case.unsupported_edges
    }
    unsupported_ratio = len(unsupported) / max(1, len(graph.edges))
    missing_context = {
        name
        for case in (plan.satisfying, plan.violating)
        for name in case.missing_context
    }
    missing_context_ratio = len(missing_context) / max(1, len(graph.nodes))
    return {
        "pair_cost": -0.8 * pair_cost,
        "unsupported": -4.0 * unsupported_ratio,
        "missing_context": -2.0 * missing_context_ratio,
    }
