from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from configfuzz.dependencies import (
    DependencyGraph,
    DependencyNodeKind,
    edge_scope_matches,
)
from configfuzz.experiment import ExperimentMethod, MutationIntent
from configfuzz.graph_solver import SolveStatus, normalize_context, solve_graph_mutation


class CampaignCaseStatus(str, Enum):
    READY = "ready"
    FILTERED = "filtered"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CampaignWorkload:
    workload_id: str
    baseline_id: str
    baseline_config: Path
    dependency_graph: Path
    static_dependency_graph: Path | None = None
    semantic_anchors: tuple[str, ...] = ()
    native_validator_manifest: Path | None = None

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, base_dir: Path
    ) -> "CampaignWorkload":
        workload_id = _required_string(data, "workload_id")
        baseline_id = _required_string(data, "baseline_id")
        baseline_config = _resolve_existing_path(
            data.get("baseline_config"), base_dir, workload_id, "baseline_config"
        )
        dependency_graph = _resolve_existing_path(
            data.get("dependency_graph"), base_dir, workload_id, "dependency_graph"
        )
        validator = data.get("native_validator_manifest")
        validator_path = None
        if validator is not None:
            validator_path = _resolve_existing_path(
                validator, base_dir, workload_id, "native_validator_manifest"
            )
        static_graph = data.get("static_dependency_graph")
        static_graph_path = None
        if static_graph is not None:
            static_graph_path = _resolve_existing_path(
                static_graph, base_dir, workload_id, "static_dependency_graph"
            )
        return cls(
            workload_id=workload_id,
            baseline_id=baseline_id,
            baseline_config=baseline_config,
            dependency_graph=dependency_graph,
            static_dependency_graph=static_graph_path,
            semantic_anchors=tuple(
                str(item) for item in data.get("semantic_anchors", ())
            ),
            native_validator_manifest=validator_path,
        )


@dataclass(frozen=True, slots=True)
class CampaignCase:
    case_id: str
    workload_id: str
    baseline_id: str
    intent_id: str
    intent_pool: str
    method: ExperimentMethod
    target_parameter: str
    target_value: Any
    status: CampaignCaseStatus
    assignments: tuple[tuple[str, Any], ...]
    coordinated_parameters: tuple[str, ...]
    target_value_preserved: bool
    preflight: str
    violated_constraints: tuple[str, ...] = ()
    unknown_constraints: tuple[str, ...] = ()
    solver_status: str | None = None
    solver_seconds: float = 0.0
    solver_timeout_ms: int | None = None
    solver_seconds_recorded_at_runtime: bool = False
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "workload_id": self.workload_id,
            "baseline_id": self.baseline_id,
            "intent_id": self.intent_id,
            "intent_pool": self.intent_pool,
            "method": self.method.value,
            "target_parameter": self.target_parameter,
            "target_value": self.target_value,
            "status": self.status.value,
            "assignments": dict(self.assignments),
            "coordinated_parameters": list(self.coordinated_parameters),
            "target_value_preserved": self.target_value_preserved,
            "preflight": self.preflight,
            "violated_constraints": list(self.violated_constraints),
            "unknown_constraints": list(self.unknown_constraints),
            "solver_status": self.solver_status,
            "solver_seconds": self.solver_seconds,
            "solver_timeout_ms": self.solver_timeout_ms,
            "solver_seconds_recorded_at_runtime": self.solver_seconds_recorded_at_runtime,
            "metadata": dict(self.metadata or {}),
        }


def load_campaign_workloads(
    path: str | Path, *, skip_unbound: bool = False
) -> dict[str, CampaignWorkload]:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("workload registry root must be an object")
    result: dict[str, CampaignWorkload] = {}
    for item in raw.get("workloads", ()):
        if not isinstance(item, Mapping):
            raise ValueError("workload entry must be an object")
        if skip_unbound and (
            item.get("baseline_config") is None or item.get("dependency_graph") is None
        ):
            continue
        workload = CampaignWorkload.from_dict(item, base_dir=source.parent)
        if workload.workload_id in result:
            raise ValueError(f"duplicate workload id: {workload.workload_id}")
        result[workload.workload_id] = workload
    return result


def load_frozen_intents(path: str | Path) -> list[MutationIntent]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("frozen intent root must be an object")
    frozen = raw.get("frozen")
    if not isinstance(frozen, Mapping):
        raise ValueError("intent set is not frozen")
    payload = {key: value for key, value in raw.items() if key != "frozen"}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected = str(frozen.get("sha256", ""))
    if actual != expected:
        raise ValueError(
            f"frozen intent hash mismatch: expected {expected}, got {actual}"
        )
    intents = [
        MutationIntent.from_dict(item)
        for item in raw.get("intents", ())
        if isinstance(item, Mapping)
    ]
    if int(frozen.get("intent_count", -1)) != len(intents):
        raise ValueError("frozen intent count mismatch")
    return intents


def plan_campaign(
    workloads: Mapping[str, CampaignWorkload],
    intents: Sequence[MutationIntent],
    *,
    methods: Sequence[ExperimentMethod] = (
        ExperimentMethod.RAW_MUTATION,
        ExperimentMethod.NATIVE_VALIDATOR_GUIDED,
        ExperimentMethod.CONSTRAINT_FILTER_ONLY,
        ExperimentMethod.STATIC_HARD_CONFIGFUZZ,
        ExperimentMethod.CONFIGFUZZ,
        ExperimentMethod.GLOBAL_REPAIR,
    ),
    solver_timeout_ms: int = 1000,
) -> dict[str, Any]:
    if solver_timeout_ms <= 0:
        raise ValueError("campaign solver timeout must be positive")
    cache: dict[
        str, tuple[Mapping[str, Any], DependencyGraph, DependencyGraph]
    ] = {}
    cases: list[CampaignCase] = []
    for intent in intents:
        workload = workloads.get(intent.workload_id)
        if workload is None:
            raise KeyError(f"intent references unknown workload: {intent.workload_id}")
        if intent.baseline_id != workload.baseline_id:
            raise ValueError(
                f"intent {intent.intent_id}: baseline {intent.baseline_id!r} does not match "
                f"workload baseline {workload.baseline_id!r}"
            )
        if workload.workload_id not in cache:
            cache[workload.workload_id] = (
                _load_json_object(workload.baseline_config),
                _load_dependency_graph(workload.dependency_graph),
                _load_dependency_graph(
                    workload.static_dependency_graph or workload.dependency_graph
                ),
            )
        baseline, graph, static_graph = cache[workload.workload_id]
        for method in methods:
            method_graph = (
                static_graph
                if method is ExperimentMethod.STATIC_HARD_CONFIGFUZZ
                else graph
            )
            cases.append(
                _plan_case(
                    workload,
                    baseline,
                    method_graph,
                    intent,
                    method,
                    solver_timeout_ms=solver_timeout_ms,
                )
            )
    method_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for case in cases:
        method_counts[case.method.value] = method_counts.get(case.method.value, 0) + 1
        status_counts[case.status.value] = status_counts.get(case.status.value, 0) + 1
    return {
        "schema_version": 1,
        "name": "configfuzz-rq2-campaign-plan",
        "intent_count": len(intents),
        "case_count": len(cases),
        "method_counts": dict(sorted(method_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "methods": [item.value for item in methods],
        "solver_timeout_ms": solver_timeout_ms,
        "cases": [case.to_dict() for case in cases],
    }


def _plan_case(
    workload: CampaignWorkload,
    baseline: Mapping[str, Any],
    graph: DependencyGraph,
    intent: MutationIntent,
    method: ExperimentMethod,
    *,
    solver_timeout_ms: int,
) -> CampaignCase:
    graph_target = _resolve_graph_parameter(graph, intent.target_parameter)
    target_assignment = ((graph_target, intent.target_value),)
    common = {
        "case_id": _case_id(intent.intent_id, method),
        "workload_id": workload.workload_id,
        "baseline_id": workload.baseline_id,
        "intent_id": intent.intent_id,
        "intent_pool": intent.intent_pool,
        "method": method,
        "target_parameter": intent.target_parameter,
        "target_value": intent.target_value,
    }
    if method is ExperimentMethod.RAW_MUTATION:
        return CampaignCase(
            **common,
            status=CampaignCaseStatus.READY,
            assignments=target_assignment,
            coordinated_parameters=(),
            target_value_preserved=True,
            preflight="none",
            metadata={
                "comparison_role": "single_parameter_mutation",
                "resolved_target_parameter": graph_target,
                "baseline_value": intent.metadata.get("baseline_value"),
                "intent_class": intent.intent_class,
            },
        )
    if method is ExperimentMethod.NATIVE_VALIDATOR_GUIDED:
        validator_bound = workload.native_validator_manifest is not None
        return CampaignCase(
            **common,
            status=(
                CampaignCaseStatus.READY
                if validator_bound
                else CampaignCaseStatus.UNKNOWN
            ),
            assignments=target_assignment,
            coordinated_parameters=(),
            target_value_preserved=True,
            preflight="native_validator",
            metadata={
                "native_validator_manifest": (
                    str(workload.native_validator_manifest) if validator_bound else None
                ),
                "reason": None
                if validator_bound
                else "native validator manifest is not bound",
                "resolved_target_parameter": graph_target,
                "baseline_value": intent.metadata.get("baseline_value"),
                "intent_class": intent.intent_class,
            },
        )
    if graph_target not in graph.nodes:
        return CampaignCase(
            **common,
            status=CampaignCaseStatus.READY,
            assignments=target_assignment,
            coordinated_parameters=(),
            target_value_preserved=True,
            preflight="no_applicable_recovered_constraint",
            unknown_constraints=(),
            metadata={
                "reason": "target parameter absent from dependency graph; preserve the requested mutation unchanged",
                "resolved_target_parameter": graph_target,
                "baseline_value": intent.metadata.get("baseline_value"),
                "intent_class": intent.intent_class,
                "repair_scope": "empty_affected_region",
            },
        )
    if method is ExperimentMethod.CONSTRAINT_FILTER_ONLY:
        context = normalize_context(graph, baseline)
        context[graph_target] = intent.target_value
        violated: list[str] = []
        unknown: list[str] = []
        for edge in sorted(graph.constraints_for(graph_target), key=lambda item: item.id):
            if not edge_scope_matches(edge, context):
                continue
            evaluation = graph.evaluate_edge(edge, context)
            if evaluation.active is False:
                continue
            if evaluation.satisfied is False:
                violated.append(edge.id)
            elif evaluation.active is None or evaluation.satisfied is None:
                unknown.append(edge.id)
        status = (
            CampaignCaseStatus.FILTERED
            if violated
            else CampaignCaseStatus.UNKNOWN
            if unknown
            else CampaignCaseStatus.READY
        )
        return CampaignCase(
            **common,
            status=status,
            assignments=target_assignment,
            coordinated_parameters=(),
            target_value_preserved=True,
            preflight="manual_constraints",
            violated_constraints=tuple(violated),
            unknown_constraints=tuple(unknown),
            metadata={
                "filter_only": True,
                "resolved_target_parameter": graph_target,
                "baseline_value": intent.metadata.get("baseline_value"),
                "intent_class": intent.intent_class,
            },
        )
    all_mutable = None
    if method is ExperimentMethod.GLOBAL_REPAIR:
        all_mutable = tuple(
            name
            for name, node in graph.nodes.items()
            if node.kind in {DependencyNodeKind.PARAMETER, DependencyNodeKind.FEATURE}
        )
    solver_started = time.perf_counter()
    semantic_anchors = tuple(
        graph_target if name == "target_parameter" else name
        for name in workload.semantic_anchors
    )
    plan = solve_graph_mutation(
        graph,
        baseline,
        graph_target,
        intent.target_value,
        static_as_hard=(method is ExperimentMethod.STATIC_HARD_CONFIGFUZZ),
        semantic_anchors=semantic_anchors,
        mutable_parameters=all_mutable,
        timeout_ms=solver_timeout_ms,
    )
    solver_seconds = time.perf_counter() - solver_started
    status = {
        SolveStatus.SAT: CampaignCaseStatus.READY,
        SolveStatus.UNSAT: CampaignCaseStatus.UNSAT,
        SolveStatus.UNKNOWN: CampaignCaseStatus.UNKNOWN,
    }[plan.status]
    assignments = (
        tuple(plan.proposed_changes)
        if plan.status is SolveStatus.SAT
        else target_assignment
    )
    assignment_map = dict(assignments)
    target_preserved = (
        assignment_map.get(graph_target) == intent.target_value
    )
    coordinated = tuple(
        name for name, _ in assignments if name != graph_target
    )
    return CampaignCase(
        **common,
        status=status,
        assignments=assignments,
        coordinated_parameters=coordinated,
        target_value_preserved=target_preserved,
        preflight="manual_constraints_and_solver",
        violated_constraints=tuple(plan.violated_soft_edges),
        unknown_constraints=tuple(plan.unsupported_edges),
        solver_status=plan.status.value,
        solver_seconds=solver_seconds,
        solver_timeout_ms=solver_timeout_ms,
        metadata={
            "mutable_parameters": list(plan.mutable_parameters),
            "compiled_edges": list(plan.compiled_edges),
            "hard_edges": list(plan.hard_edges),
            "out_of_scope_edges": list(plan.out_of_scope_edges),
            "missing_context": list(plan.missing_context),
            "reason": plan.reason,
            "resolved_target_parameter": graph_target,
            "baseline_value": intent.metadata.get("baseline_value"),
            "intent_class": intent.intent_class,
            "repair_scope": (
                "all_parameters"
                if method is ExperimentMethod.GLOBAL_REPAIR
                else "affected_hypergraph_region"
            ),
            "constraint_treatment": (
                "all_static_candidates_hard"
                if method is ExperimentMethod.STATIC_HARD_CONFIGFUZZ
                else "status_and_confidence_aware"
            ),
            "constraint_graph_source": (
                "pre_validation_static_graph"
                if method is ExperimentMethod.STATIC_HARD_CONFIGFUZZ
                and workload.static_dependency_graph is not None
                else "validated_graph"
            ),
        },
    )


def _resolve_graph_parameter(graph: DependencyGraph, parameter: str) -> str:
    if parameter in graph.nodes:
        return parameter
    leaf = parameter.rsplit(".", 1)[-1]
    if leaf in graph.nodes:
        return leaf
    matches = [name for name in graph.nodes if name.rsplit(".", 1)[-1] == leaf]
    if len(matches) == 1:
        return matches[0]
    return parameter



def _case_id(intent_id: str, method: ExperimentMethod) -> str:
    digest = hashlib.sha256(f"{intent_id}:{method.value}".encode("utf-8")).hexdigest()[
        :12
    ]
    return f"{intent_id}-{method.value}-{digest}"


def _load_json_object(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return raw


def _load_dependency_graph(path: Path) -> DependencyGraph:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"dependency graph object required: {path}")
    return DependencyGraph.from_dict(raw)


def _resolve_existing_path(
    value: Any,
    base_dir: Path,
    workload_id: str,
    field: str,
) -> Path:
    if value is None:
        raise ValueError(f"workload {workload_id}: {field} is not bound")
    path = Path(str(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"workload {workload_id}: {field} not found: {path}")
    return path


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{key} must be a non-empty string")
    return str(value).strip()
