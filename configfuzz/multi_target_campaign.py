from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from configfuzz.dependencies import DependencyGraph
from configfuzz.experiment import ExperimentMethod, MutationIntent
from configfuzz.experiment_campaign import CampaignWorkload, load_campaign_workloads
from configfuzz.graph_solver import SolveStatus, solve_graph_mutations
from configfuzz.rq2_gpu_executor import (
    RUNTIME_INSTRUMENTATION_VERSION,
    _available_master_port,
    execute_campaign_case,
)


@dataclass(frozen=True, slots=True)
class BoundIntent:
    intent: MutationIntent
    assignment_parameter: str
    graph_parameter: str | None


def load_candidate_intents(path: str | Path) -> list[MutationIntent]:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("intents"), list):
        raise ValueError("candidate intent file must contain an intents array")
    return [
        MutationIntent.from_dict(item)
        for item in raw["intents"]
        if isinstance(item, Mapping)
    ]


def build_candidate_pools(
    workloads: Mapping[str, CampaignWorkload],
    intents: Sequence[MutationIntent],
    *,
    intent_pool: str = "method_independent",
) -> tuple[dict[str, dict[str, tuple[BoundIntent, ...]]], dict[str, DependencyGraph]]:
    graphs = {
        workload_id: _load_graph(workload.dependency_graph)
        for workload_id, workload in workloads.items()
    }
    grouped: dict[str, dict[str, list[BoundIntent]]] = {
        workload_id: {} for workload_id in workloads
    }
    for intent in intents:
        workload = workloads.get(intent.workload_id)
        if workload is None or intent.intent_pool != intent_pool:
            continue
        if intent.baseline_id != workload.baseline_id:
            continue
        graph = graphs[intent.workload_id]
        graph_parameter = _resolve_graph_parameter(graph, intent.target_parameter)
        assignment_parameter = graph_parameter or intent.target_parameter
        grouped[intent.workload_id].setdefault(assignment_parameter, []).append(
            BoundIntent(
                intent=intent,
                assignment_parameter=assignment_parameter,
                graph_parameter=graph_parameter,
            )
        )

    pools: dict[str, dict[str, tuple[BoundIntent, ...]]] = {}
    for workload_id, parameter_groups in grouped.items():
        pools[workload_id] = {
            parameter: tuple(sorted(items, key=lambda item: item.intent.intent_id))
            for parameter, items in sorted(parameter_groups.items())
        }
    return pools, graphs


def plan_multi_target_round(
    *,
    round_index: int,
    seed: int,
    targets_per_mutation: int,
    workloads: Mapping[str, CampaignWorkload],
    candidate_pools: Mapping[str, Mapping[str, Sequence[BoundIntent]]],
    graphs: Mapping[str, DependencyGraph],
    solver_timeout_ms: int = 1000,
) -> dict[str, Any]:
    if round_index < 1:
        raise ValueError("round_index must be positive")
    if targets_per_mutation < 2:
        raise ValueError("multi-target mutation requires --mutnm >= 2")
    if solver_timeout_ms <= 0:
        raise ValueError("solver timeout must be positive")

    eligible = sorted(
        workload_id
        for workload_id, groups in candidate_pools.items()
        if workload_id in workloads and len(groups) >= targets_per_mutation
    )
    if not eligible:
        raise ValueError(
            f"no workload has {targets_per_mutation} distinct mutation targets"
        )

    rng = random.Random(_round_seed(seed, round_index))
    workload_id = eligible[rng.randrange(len(eligible))]
    workload = workloads[workload_id]
    groups = candidate_pools[workload_id]
    parameters = rng.sample(sorted(groups), targets_per_mutation)
    selected = [
        groups[parameter][rng.randrange(len(groups[parameter]))]
        for parameter in parameters
    ]
    selected.sort(key=lambda item: item.assignment_parameter)

    target_assignments = {
        item.assignment_parameter: item.intent.target_value for item in selected
    }
    graph_assignments = {
        item.graph_parameter: item.intent.target_value
        for item in selected
        if item.graph_parameter is not None
    }
    raw_assignments = {
        item.assignment_parameter: item.intent.target_value
        for item in selected
        if item.graph_parameter is None
    }

    baseline = _load_json_object(workload.baseline_config)
    solver_started = time.perf_counter()
    plan = None
    if graph_assignments:
        anchors = tuple(
            name for name in workload.semantic_anchors if name != "target_parameter"
        )
        plan = solve_graph_mutations(
            graphs[workload_id],
            baseline,
            graph_assignments,
            semantic_anchors=anchors,
            timeout_ms=solver_timeout_ms,
        )
    solver_seconds = time.perf_counter() - solver_started

    if plan is None:
        status = "ready"
        assignments = dict(raw_assignments)
        coordinated: tuple[str, ...] = ()
        violated: tuple[str, ...] = ()
        unknown: tuple[str, ...] = ()
        solver_status = "not_required"
        solver_metadata: dict[str, Any] = {}
    else:
        status = {
            SolveStatus.SAT: "ready",
            SolveStatus.UNSAT: "unsat",
            SolveStatus.UNKNOWN: "unknown",
        }[plan.status]
        assignments = dict(raw_assignments)
        if plan.status is SolveStatus.SAT:
            assignments.update(plan.changes)
        else:
            assignments.update(target_assignments)
        target_names = set(target_assignments)
        coordinated = tuple(
            sorted(name for name in assignments if name not in target_names)
        )
        violated = plan.violated_soft_edges
        unknown = plan.unsupported_edges
        solver_status = plan.status.value
        solver_metadata = {
            "mutable_parameters": list(plan.mutable_parameters),
            "compiled_edges": list(plan.compiled_edges),
            "hard_edges": list(plan.hard_edges),
            "soft_edges": list(plan.soft_edges),
            "out_of_scope_edges": list(plan.out_of_scope_edges),
            "missing_context": list(plan.missing_context),
            "reason": plan.reason,
        }

    selected_ids = [item.intent.intent_id for item in selected]
    digest = hashlib.sha256(
        json.dumps(
            {
                "round": round_index,
                "seed": seed,
                "workload": workload_id,
                "intent_ids": selected_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    intent_id = f"multi-target-{round_index:08d}-{digest}"
    return {
        "case_id": intent_id,
        "workload_id": workload_id,
        "baseline_id": workload.baseline_id,
        "intent_id": intent_id,
        "intent_pool": "multi_target",
        "method": ExperimentMethod.CONFIGFUZZ.value,
        "target_assignments": target_assignments,
        "status": status,
        "assignments": assignments,
        "coordinated_parameters": coordinated,
        "target_value_preserved": status == "ready",
        "preflight": "joint_multi_target_solver",
        "violated_constraints": violated,
        "unknown_constraints": unknown,
        "solver_status": solver_status,
        "solver_seconds": solver_seconds,
        "metadata": {
            "campaign_round": round_index,
            "campaign_seed": seed,
            "targets_per_mutation": targets_per_mutation,
            "selected_intent_ids": selected_ids,
            "selected_intent_classes": [item.intent.intent_class for item in selected],
            "selected_target_parameters": [
                item.intent.target_parameter for item in selected
            ],
            "resolved_target_parameters": [
                item.assignment_parameter for item in selected
            ],
            **solver_metadata,
        },
    }


def execute_multi_target_campaign(
    *,
    framework_id: str,
    workload_registry_path: str | Path,
    candidate_path: str | Path,
    launcher: str | Path,
    output_root: str | Path,
    output_jsonl: str | Path,
    rounds: int,
    targets_per_mutation: int,
    seed: int,
    gpu_devices: str = "4,5",
    device_count: int = 2,
    master_port: int = 30001,
    timeout_seconds: float = 120.0,
    solver_timeout_ms: int = 1000,
    workload_ids: Sequence[str] = (),
    intent_pool: str = "method_independent",
) -> dict[str, Any]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if targets_per_mutation < 2:
        raise ValueError("multi-target mutation requires --mutnm >= 2")
    if device_count <= 0:
        raise ValueError("device_count must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    workload_path = Path(workload_registry_path).expanduser().resolve()
    candidate_file = Path(candidate_path).expanduser().resolve()
    launcher_path = Path(launcher).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    output = Path(output_jsonl).expanduser().resolve()
    workloads = load_campaign_workloads(workload_path)
    selected_workloads = set(workload_ids)
    if selected_workloads:
        workloads = {
            workload_id: workload
            for workload_id, workload in workloads.items()
            if workload_id in selected_workloads
        }
        missing = selected_workloads - set(workloads)
        if missing:
            raise KeyError(f"unknown workload ids: {sorted(missing)}")
    if not workloads:
        raise ValueError("no workloads selected")

    intents = load_candidate_intents(candidate_file)
    pools, graphs = build_candidate_pools(
        workloads,
        intents,
        intent_pool=intent_pool,
    )
    insufficient = {
        workload_id: len(groups)
        for workload_id, groups in pools.items()
        if len(groups) < targets_per_mutation
    }
    if len(insufficient) == len(pools):
        raise ValueError(
            f"no selected workload has {targets_per_mutation} distinct candidate parameters: "
            f"{insufficient}"
        )

    root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids, cumulative_accelerator_seconds = _resume_state(output)
    provenance = {
        "runner_revision": _git_revision(),
        "workload_registry_sha256": _file_sha256(workload_path),
        "candidate_sha256": _file_sha256(candidate_file),
        "launcher_sha256": _file_sha256(launcher_path),
        "runtime_instrumentation": RUNTIME_INSTRUMENTATION_VERSION,
        "campaign_kind": "multi_target_continuous_v1",
    }

    campaign_started = time.monotonic()
    written = 0
    skipped = 0
    launched = 0
    planned_ready = 0
    with output.open("a", encoding="utf-8") as handle:
        for round_index in range(1, rounds + 1):
            case = plan_multi_target_round(
                round_index=round_index,
                seed=seed,
                targets_per_mutation=targets_per_mutation,
                workloads=workloads,
                candidate_pools=pools,
                graphs=graphs,
                solver_timeout_ms=solver_timeout_ms,
            )
            run_id = _run_id(framework_id, case, seed)
            if run_id in completed_ids:
                skipped += 1
                continue
            if case["status"] == "ready":
                planned_ready += 1
            workload = workloads[str(case["workload_id"])]
            record, used_accelerator = execute_campaign_case(
                framework_id=framework_id,
                case=case,
                workload=workload,
                launcher=launcher_path,
                output_root=root,
                gpu_devices=gpu_devices,
                device_count=device_count,
                accelerator_kind="gpu",
                seed=seed,
                master_port=_available_master_port(master_port + (round_index % 200)),
                timeout_seconds=timeout_seconds,
                run_id=run_id,
                campaign_test_index=round_index,
                campaign_started=campaign_started,
                cumulative_accelerator_seconds=cumulative_accelerator_seconds,
                provenance=provenance,
                rq="rq3",
            )
            cumulative_accelerator_seconds += record.gpu_seconds
            payload = record.to_dict()
            payload["campaign_gpu_seconds"] = cumulative_accelerator_seconds
            payload["campaign_accelerator_seconds"] = cumulative_accelerator_seconds
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed_ids.add(run_id)
            written += 1
            launched += int(used_accelerator)

    return {
        "framework_id": framework_id,
        "rounds": rounds,
        "targets_per_mutation": targets_per_mutation,
        "seed": seed,
        "written_records": written,
        "skipped_existing": skipped,
        "ready_plans_written": planned_ready,
        "accelerator_launches": launched,
        "candidate_intents": sum(
            len(items) for groups in pools.values() for items in groups.values()
        ),
        "candidate_parameters": sum(len(groups) for groups in pools.values()),
        "output": str(output),
    }


def _resolve_graph_parameter(graph: DependencyGraph, parameter: str) -> str | None:
    if parameter in graph.nodes:
        return parameter
    leaf = parameter.rsplit(".", 1)[-1]
    if leaf in graph.nodes:
        return leaf
    matches = [name for name in graph.nodes if name.rsplit(".", 1)[-1] == leaf]
    if len(matches) == 1:
        return matches[0]
    return None


def _round_seed(seed: int, round_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{round_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _run_id(framework_id: str, case: Mapping[str, Any], seed: int) -> str:
    material = f"rq3:{framework_id}:{case['case_id']}:{seed}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"rq3-{framework_id}-multi-{digest}"


def _resume_state(path: Path) -> tuple[set[str], float]:
    if not path.is_file():
        return set(), 0.0
    run_ids: set[str] = set()
    accelerator_seconds = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            continue
        if payload.get("run_id"):
            run_ids.add(str(payload["run_id"]))
        accelerator_seconds += float(
            payload.get("accelerator_seconds", payload.get("gpu_seconds", 0.0))
        )
    return run_ids, accelerator_seconds


def _load_graph(path: Path) -> DependencyGraph:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"dependency graph object required: {path}")
    return DependencyGraph.from_dict(raw)


def _load_json_object(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return raw


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
