from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from configfuzz.experiment import (
    ExecutionMilestone,
    ExperimentMethod,
    ExperimentOutcome,
    ExperimentRunRecord,
    MILESTONE_ORDER,
)
from configfuzz.experiment_campaign import CampaignWorkload, load_campaign_workloads


_MILESTONE_NAMES = {
    "argument_parsing": ExecutionMilestone.ARGUMENT_PARSING,
    "configuration_validation": ExecutionMilestone.CONFIG_VALIDATION,
    "config_validation": ExecutionMilestone.CONFIG_VALIDATION,
    "model_construction": ExecutionMilestone.MODEL_CONSTRUCTION,
    "distributed_initialization": ExecutionMilestone.PROCESS_GROUP_INITIALIZATION,
    "process_group_initialization": ExecutionMilestone.PROCESS_GROUP_INITIALIZATION,
    "forward": ExecutionMilestone.FORWARD,
    "backward": ExecutionMilestone.BACKWARD,
    "optimizer_step": ExecutionMilestone.OPTIMIZER_STEP,
    "repeated_training": ExecutionMilestone.REPEATED_TRAINING,
    "checkpoint_save_load": ExecutionMilestone.CHECKPOINT_SAVE_LOAD,
    "completed": ExecutionMilestone.COMPLETED,
}
_RESOURCE_PATTERNS = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)
_INFRA_PATTERNS = (
    "connection refused",
    "no route to host",
    "address already in use",
    "nccl error",
    "socket timeout",
)
_VALIDATION_PATTERNS = (
    "configfuzz_native_validation_rejected",
    "invalid choice",
    "valueerror:",
    "assertionerror:",
    "cannot use sequence parallelism",
    "must be",
    "should be",
)
_ERROR_LINE = re.compile(r"(?:ValueError|AssertionError|RuntimeError|ZeroDivisionError|TypeError):.*")
_RUNTIME_EVENT_PREFIX = "CONFIGFUZZ_RUNTIME_EVENT:"
_RUNTIME_EVENT_KINDS = {"branch", "backend", "topology", "feature"}
RUNTIME_INSTRUMENTATION_VERSION = "runtime-events-v1"


def load_plan(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("cases"), list):
        raise ValueError("RQ2 plan must contain a cases array")
    return payload


def materialize_profile(
    workload: CampaignWorkload,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    profile = json.loads(workload.baseline_config.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError(f"baseline must be a JSON object: {workload.baseline_config}")
    assignments = case.get("assignments", {})
    if not isinstance(assignments, Mapping):
        raise ValueError("case assignments must be an object")
    for parameter, value in assignments.items():
        _assign(profile, str(parameter), copy.deepcopy(value))
    return profile


def execute_primary_campaign(
    *,
    framework_id: str,
    plan_path: str | Path,
    workload_registry_path: str | Path,
    launcher: str | Path,
    output_root: str | Path,
    output_jsonl: str | Path,
    gpu_devices: str = "4,5",
    device_count: int = 2,
    seed: int = 2026,
    master_port: int = 30001,
    timeout_seconds: float = 120.0,
    workload_ids: Sequence[str] = (),
    methods: Sequence[ExperimentMethod] = (),
    intent_ids: Sequence[str] = (),
    limit: int | None = None,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    workloads = load_campaign_workloads(workload_registry_path)
    launcher_path = Path(launcher).resolve()
    root = Path(output_root).resolve()
    output = Path(output_jsonl).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "runner_revision": _git_revision(),
        "plan_sha256": _file_sha256(Path(plan_path)),
        "workload_registry_sha256": _file_sha256(Path(workload_registry_path)),
        "launcher_sha256": _file_sha256(launcher_path),
        "runtime_instrumentation": RUNTIME_INSTRUMENTATION_VERSION,
    }

    completed_ids = _existing_run_ids(output)
    selected_workloads = set(workload_ids)
    selected_methods = set(methods)
    selected_intents = set(intent_ids)
    cases: list[Mapping[str, Any]] = []
    for raw in plan["cases"]:
        if not isinstance(raw, Mapping):
            continue
        if selected_workloads and str(raw.get("workload_id")) not in selected_workloads:
            continue
        method = ExperimentMethod(str(raw["method"]))
        if selected_methods and method not in selected_methods:
            continue
        if selected_intents and str(raw.get("intent_id")) not in selected_intents:
            continue
        cases.append(raw)
    if limit is not None:
        cases = cases[:limit]

    campaign_started = time.monotonic()
    cumulative_accelerator_seconds = 0.0
    written = 0
    skipped = 0
    launched = 0
    with output.open("a", encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            run_id = _run_id(framework_id, case, seed)
            if run_id in completed_ids:
                skipped += 1
                continue
            workload_id = str(case["workload_id"])
            workload = workloads.get(workload_id)
            if workload is None:
                raise KeyError(f"plan references workload not supported by {framework_id}: {workload_id}")
            record, used_accelerator = _execute_case(
                framework_id=framework_id,
                case=case,
                workload=workload,
                launcher=launcher_path,
                output_root=root,
                gpu_devices=gpu_devices,
                device_count=device_count,
                seed=seed,
                master_port=_available_master_port(master_port + (index % 200)),
                timeout_seconds=timeout_seconds,
                run_id=run_id,
                campaign_test_index=index,
                campaign_started=campaign_started,
                cumulative_accelerator_seconds=cumulative_accelerator_seconds,
                provenance=provenance,
            )
            cumulative_accelerator_seconds += record.gpu_seconds
            payload = record.to_dict()
            payload["campaign_gpu_seconds"] = cumulative_accelerator_seconds
            payload["campaign_accelerator_seconds"] = cumulative_accelerator_seconds
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
            launched += int(used_accelerator)
    return {
        "framework_id": framework_id,
        "selected_cases": len(cases),
        "written_records": written,
        "skipped_existing": skipped,
        "accelerator_launches": launched,
        "output": str(output),
    }


def _execute_case(
    *,
    framework_id: str,
    case: Mapping[str, Any],
    workload: CampaignWorkload,
    launcher: Path,
    output_root: Path,
    gpu_devices: str,
    device_count: int,
    seed: int,
    master_port: int,
    timeout_seconds: float,
    run_id: str,
    campaign_test_index: int,
    campaign_started: float,
    cumulative_accelerator_seconds: float,
    provenance: Mapping[str, str],
) -> tuple[ExperimentRunRecord, bool]:
    method = ExperimentMethod(str(case["method"]))
    status = str(case.get("status", "unknown"))
    common = {
        "run_id": run_id,
        "rq": "rq2",
        "method": method,
        "workload_id": str(case["workload_id"]),
        "baseline_id": str(case["baseline_id"]),
        "intent_id": str(case["intent_id"]),
        "intent_pool": str(case.get("intent_pool", "method_independent")),
        "seed": seed,
        "target_value_preserved": bool(case.get("target_value_preserved", False)),
        "coordinated_parameters": tuple(str(x) for x in case.get("coordinated_parameters", ())),
        "modification_distance": _modification_distance(case),
        "solver_seconds": float(case.get("solver_seconds", 0.0)),
        "peak_memory_mib": None,
        "campaign_test_index": campaign_test_index,
        "constraints_exercised": tuple(str(x) for x in case.get("violated_constraints", ())),
        "boundaries_exercised": (f"{case.get('target_parameter')}={case.get('target_value')!r}",),
        "affected_region": tuple(str(x) for x in case.get("coordinated_parameters", ())),
        "solver_modifications": dict(case.get("assignments", {})) if isinstance(case.get("assignments"), Mapping) else {},
        "metadata": {
            "framework_id": framework_id,
            "planner_status": status,
            "preflight": case.get("preflight"),
            "target_parameter": case.get("target_parameter"),
            "target_value": case.get("target_value"),
            "case_metadata": dict(case.get("metadata", {})) if isinstance(case.get("metadata"), Mapping) else {},
            **dict(provenance),
        },
    }
    if status != "ready":
        outcome = ExperimentOutcome.EXPECTED_REJECTION if status in {"filtered", "unsat"} else ExperimentOutcome.UNKNOWN
        return (
            ExperimentRunRecord(
                **common,
                generated=False,
                deepest_milestone=ExecutionMilestone.CONFIG_GENERATION,
                outcome=outcome,
                duration_seconds=0.0,
                gpu_seconds=0.0,
                timed_out=False,
                campaign_elapsed_seconds=time.monotonic() - campaign_started,
                campaign_gpu_seconds=cumulative_accelerator_seconds,
            ),
            False,
        )

    case_root = output_root / run_id
    case_root.mkdir(parents=True, exist_ok=True)
    config_path = case_root / "config.json"
    log_path = case_root / "run.log"
    try:
        profile = materialize_profile(workload, case)
    except (KeyError, ValueError, TypeError) as exc:
        metadata = dict(common["metadata"])
        metadata["materialization_error"] = str(exc)
        common["metadata"] = metadata
        return (
            ExperimentRunRecord(
                **common,
                generated=False,
                deepest_milestone=ExecutionMilestone.CONFIG_GENERATION,
                outcome=ExperimentOutcome.UNKNOWN,
                duration_seconds=0.0,
                gpu_seconds=0.0,
                timed_out=False,
                campaign_elapsed_seconds=time.monotonic() - campaign_started,
                campaign_gpu_seconds=cumulative_accelerator_seconds,
            ),
            False,
        )
    config_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "CONFIGFUZZ_GPU_DEVICES": gpu_devices,
            "CONFIGFUZZ_NPROC_PER_NODE": str(device_count),
            "CONFIGFUZZ_MASTER_PORT": str(master_port),
            "CONFIGFUZZ_SEED": str(seed),
            "CONFIGFUZZ_RQ2_SKIP_CHECKPOINT": "1",
            "CONFIGFUZZ_CHECKPOINT_ROOT": str(case_root / "checkpoints"),
        }
    )
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            ["bash", str(launcher), str(config_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        text = completed.stdout or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        text = exc.stdout if isinstance(exc.stdout, str) else ""
    duration = time.monotonic() - started
    log_path.write_text(text, encoding="utf-8", errors="replace")
    milestone = deepest_milestone(text, timed_out)
    outcome = classify_outcome(text, returncode, timed_out, milestone)
    metadata = dict(common["metadata"])
    metadata.update(
        {
            "returncode": returncode,
            "config_path": str(config_path),
            "log_path": str(log_path),
            "gpu_devices": gpu_devices,
            "device_count": device_count,
        }
    )
    common["metadata"] = metadata
    runtime_events = parse_runtime_events(text)
    record = ExperimentRunRecord(
        **common,
        generated=True,
        deepest_milestone=milestone,
        outcome=outcome,
        duration_seconds=duration,
        gpu_seconds=duration * device_count,
        timed_out=timed_out,
        campaign_elapsed_seconds=time.monotonic() - campaign_started,
        campaign_gpu_seconds=cumulative_accelerator_seconds + duration * device_count,
        runtime_branches=runtime_events["branch"],
        topologies=runtime_events["topology"],
        feature_interactions=runtime_events["feature"],
        backend_paths=runtime_events["backend"],
        behavior_ids=runtime_events["behavior_ids"],
        behavior_signature=runtime_events["behavior_signature"],
    )
    return record, True


def deepest_milestone(output: str, timed_out: bool) -> ExecutionMilestone:
    if timed_out:
        return ExecutionMilestone.TIMEOUT
    seen = [
        _MILESTONE_NAMES[name]
        for name in re.findall(r"CONFIGFUZZ_MILESTONE:([a-z_]+)", output)
        if name in _MILESTONE_NAMES
    ]
    if not seen:
        return ExecutionMilestone.UNKNOWN
    return max(seen, key=lambda item: MILESTONE_ORDER[item])


def classify_outcome(
    output: str,
    returncode: int,
    timed_out: bool,
    milestone: ExecutionMilestone,
) -> ExperimentOutcome:
    lowered = output.lower()
    if timed_out:
        return ExperimentOutcome.UNEXPLAINED_FAILURE
    if returncode == 0:
        return ExperimentOutcome.VALID
    if any(pattern in lowered for pattern in _RESOURCE_PATTERNS):
        return ExperimentOutcome.RESOURCE_FAILURE
    if any(pattern in lowered for pattern in _INFRA_PATTERNS):
        return ExperimentOutcome.INFRASTRUCTURE_FAILURE
    reached_forward = MILESTONE_ORDER.get(milestone, -1) >= MILESTONE_ORDER[ExecutionMilestone.FORWARD]
    if not reached_forward and any(pattern in lowered for pattern in _VALIDATION_PATTERNS):
        return ExperimentOutcome.EXPECTED_REJECTION
    return ExperimentOutcome.UNEXPLAINED_FAILURE


def behavior_signature(
    output: str,
    milestone: ExecutionMilestone,
    outcome: ExperimentOutcome,
) -> str:
    errors = _ERROR_LINE.findall(output)
    tail = errors[-1].strip() if errors else ""
    material = f"{milestone.value}\n{outcome.value}\n{tail}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def parse_runtime_events(output: str) -> dict[str, Any]:
    categorized: dict[str, set[str]] = {kind: set() for kind in _RUNTIME_EVENT_KINDS}
    for line in output.splitlines():
        if not line.startswith(_RUNTIME_EVENT_PREFIX):
            continue
        raw = line[len(_RUNTIME_EVENT_PREFIX) :]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        kind = str(payload.get("kind", ""))
        value = payload.get("value")
        if kind not in _RUNTIME_EVENT_KINDS or not isinstance(value, str) or not value:
            continue
        categorized[kind].add(value)

    behavior_ids = tuple(
        sorted(
            f"{kind}:{value}"
            for kind, values in categorized.items()
            for value in values
        )
    )
    signature = None
    if behavior_ids:
        material = json.dumps(behavior_ids, ensure_ascii=False, separators=(",", ":"))
        signature = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return {
        **{kind: tuple(sorted(values)) for kind, values in categorized.items()},
        "behavior_ids": behavior_ids,
        "behavior_signature": signature,
    }


def _assign(profile: dict[str, Any], parameter: str, value: Any) -> None:
    parts = parameter.split(".")
    current: Any = profile
    exact = True
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            exact = False
            break
        current = current[part]
    if exact and isinstance(current, dict) and parts[-1] in current:
        current[parts[-1]] = value
        return

    leaf = parts[-1]
    matches: list[tuple[dict[str, Any], str]] = []

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, item in node.items():
            if key == leaf and not isinstance(item, (dict, list)):
                matches.append((node, key))
            if isinstance(item, dict):
                visit(item)

    visit(profile)
    if len(matches) != 1:
        raise KeyError(f"cannot uniquely bind assignment {parameter!r}; matches={len(matches)}")
    matches[0][0][matches[0][1]] = value


def _modification_distance(case: Mapping[str, Any]) -> float | None:
    metadata = case.get("metadata")
    assignments = case.get("assignments")
    if not isinstance(metadata, Mapping) or not isinstance(assignments, Mapping):
        return None
    baseline = metadata.get("baseline_value")
    target = str(metadata.get("resolved_target_parameter", case.get("target_parameter", "")))
    if baseline is None or target not in assignments:
        return None
    value = assignments[target]
    if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
        return abs(float(value) - float(baseline))
    return 0.0 if value == baseline else 1.0


def _run_id(framework_id: str, case: Mapping[str, Any], seed: int) -> str:
    material = f"{framework_id}:{case['case_id']}:{seed}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"rq2-{framework_id}-{digest}"


def _available_master_port(preferred: int, *, attempts: int = 256) -> int:
    """Return an unused localhost TCP port near ``preferred``.

    RQ2 cases execute sequentially, but other experiments on the same worker can
    occupy ports in the campaign's nominal range.  Probing before each launch
    prevents a transient ``EADDRINUSE`` from being recorded as a framework
    outcome.
    """

    if not 1 <= preferred <= 65535:
        raise ValueError("preferred port must be in [1, 65535]")
    for offset in range(attempts):
        port = preferred + offset
        if port > 65535:
            port = 1024 + (port - 65536)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError(
        f"could not find a free master port near {preferred} after {attempts} attempts"
    )


def _existing_run_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    run_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping) and payload.get("run_id"):
            run_ids.add(str(payload["run_id"]))
    return run_ids


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
