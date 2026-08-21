from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from configfuzz.experiment import (
    ExecutionMilestone,
    ExperimentMethod,
    ExperimentOutcome,
    ExperimentRunRecord,
)


@dataclass(frozen=True, slots=True)
class Qwen2PilotRuntime:
    launcher: Path
    output_root: Path
    device: int = 0
    master_port: int = 6300
    timeout_seconds: float = 180.0
    seed: int = 42


# Pilot mappings are intentionally explicit. The formal campaign must bind each
# workload to its own command manifest rather than guessing CLI spellings.
_VALUE_FLAGS: dict[str, str] = {
    "attention_dropout": "--attention-dropout",
    "ffn_hidden_size": "--ffn-hidden-size",
    "global_batch_size": "--global-batch-size",
    "hidden_dropout": "--hidden-dropout",
    "hidden_size": "--hidden-size",
    "init_method_std": "--init-method-std",
    "lr_decay_iters": "--lr-decay-iters",
    "lr_warmup_iters": "--lr-warmup-iters",
    "make_vocab_size_divisible_by": "--make-vocab-size-divisible-by",
    "max_position_embeddings": "--max-position-embeddings",
    "micro_batch_size": "--micro-batch-size",
    "num_attention_heads": "--num-attention-heads",
    "seq_length": "--seq-length",
    "train_iters": "--train-iters",
    "vocab_size": "--vocab-size",
}

_TRUE_FLAGS: dict[str, str] = {
    "attention_softmax_in_fp32": "--attention-softmax-in-fp32",
    "group_query_attention": "--group-query-attention",
    "sequence_parallel": "--sequence-parallel",
    "swiglu": "--swiglu",
    "untie_embeddings_and_output_weights": "--untie-embeddings-and-output-weights",
}

_INFRA_PATTERNS = (
    "HCCP_Process_Initialization_Failure",
    "Failed to initialize the HCCP process",
    "Connection refused",
    "No route to host",
)
_RESOURCE_PATTERNS = (
    "out of memory",
    "NPU out of memory",
    "alloc device memory failed",
    "ACL_ERROR_RT_MEMORY_ALLOCATION",
)


def load_campaign_plan(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("cases"), list):
        raise ValueError("RQ2 campaign plan must contain a cases array")
    return payload


def execute_qwen2_pilot_cases(
    plan: Mapping[str, Any],
    runtime: Qwen2PilotRuntime,
    *,
    intent_ids: Sequence[str] = (),
    methods: Sequence[ExperimentMethod] = (),
) -> list[ExperimentRunRecord]:
    selected_intents = set(intent_ids)
    selected_methods = set(methods)
    cases: list[Mapping[str, Any]] = []
    for raw in plan.get("cases", ()):
        if not isinstance(raw, Mapping):
            continue
        if raw.get("workload_id") != "qwen2-text-train":
            continue
        if selected_intents and str(raw.get("intent_id")) not in selected_intents:
            continue
        method = ExperimentMethod(str(raw["method"]))
        if selected_methods and method not in selected_methods:
            continue
        cases.append(raw)

    records: list[ExperimentRunRecord] = []
    runtime_cache: dict[tuple[str, str, tuple[str, ...]], ExperimentRunRecord] = {}
    campaign_started = time.monotonic()
    cumulative_accelerator_seconds = 0.0
    for index, case in enumerate(cases, 1):
        runtime_key = _pilot_runtime_key(case)
        cached = runtime_cache.get(runtime_key) if runtime_key is not None else None
        if cached is not None:
            records.append(
                _reuse_pilot_runtime_record(
                    cached,
                    case,
                    campaign_test_index=index,
                    campaign_elapsed_seconds=time.monotonic() - campaign_started,
                    campaign_accelerator_seconds=cumulative_accelerator_seconds,
                )
            )
            continue

        record = execute_qwen2_pilot_case(
            case,
            runtime,
            campaign_test_index=index,
            campaign_started=campaign_started,
            cumulative_gpu_seconds=cumulative_accelerator_seconds,
        )
        cumulative_accelerator_seconds += record.accelerator_seconds
        payload = record.to_dict()
        payload["campaign_gpu_seconds"] = cumulative_accelerator_seconds
        payload["campaign_accelerator_seconds"] = cumulative_accelerator_seconds
        record = ExperimentRunRecord.from_dict(payload)
        records.append(record)
        if (
            runtime_key is not None
            and record.generated
            and record.outcome is not ExperimentOutcome.INFRASTRUCTURE_FAILURE
        ):
            runtime_cache[runtime_key] = record
    return records


def _pilot_runtime_key(
    case: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...]] | None:
    if str(case.get("status", "unknown")) != "ready":
        return None
    assignments = case.get("assignments")
    if not isinstance(assignments, Mapping):
        return None
    try:
        cli_args = assignments_to_qwen2_cli(assignments)
    except (KeyError, ValueError, TypeError):
        return None
    return (
        str(case.get("workload_id", "")),
        str(case.get("baseline_id", "")),
        tuple(cli_args),
    )


def _reuse_pilot_runtime_record(
    source: ExperimentRunRecord,
    case: Mapping[str, Any],
    *,
    campaign_test_index: int,
    campaign_elapsed_seconds: float,
    campaign_accelerator_seconds: float,
) -> ExperimentRunRecord:
    payload = source.to_dict()
    assignments = case.get("assignments")
    source_metadata = dict(source.metadata)
    payload.update(
        {
            "run_id": f"rq2-pilot-{case['case_id']}",
            "method": str(case["method"]),
            "workload_id": str(case["workload_id"]),
            "baseline_id": str(case["baseline_id"]),
            "intent_id": str(case["intent_id"]),
            "target_value_preserved": bool(case.get("target_value_preserved", False)),
            "coordinated_parameters": list(case.get("coordinated_parameters", ())),
            "modification_distance": _modification_distance(case),
            "solver_seconds": float(case.get("solver_seconds", 0.0)),
            "campaign_test_index": campaign_test_index,
            "campaign_elapsed_seconds": campaign_elapsed_seconds,
            "campaign_gpu_seconds": campaign_accelerator_seconds,
            "campaign_accelerator_seconds": campaign_accelerator_seconds,
            "constraints_exercised": list(case.get("violated_constraints", ())),
            "boundaries_exercised": [_boundary_label(case)],
            "solver_modifications": dict(assignments) if isinstance(assignments, Mapping) else {},
            "metadata": {
                **source_metadata,
                "planner_status": str(case.get("status", "unknown")),
                "preflight": case.get("preflight"),
                "runtime_reuse": {
                    "source_run_id": source.run_id,
                    "source_method": source.method.value,
                    "source_intent_id": source.intent_id,
                    "rationale": (
                        "same workload and baseline with byte-equivalent effective "
                        "Ascend launcher CLI under the same pilot runtime"
                    ),
                },
            },
        }
    )
    return ExperimentRunRecord.from_dict(payload)


def execute_qwen2_pilot_case(
    case: Mapping[str, Any],
    runtime: Qwen2PilotRuntime,
    *,
    campaign_test_index: int,
    campaign_started: float,
    cumulative_gpu_seconds: float,
) -> ExperimentRunRecord:
    method = ExperimentMethod(str(case["method"]))
    status = str(case.get("status", "unknown"))
    common = {
        "run_id": f"rq2-pilot-{case['case_id']}",
        "rq": "rq2",
        "method": method,
        "workload_id": str(case["workload_id"]),
        "baseline_id": str(case["baseline_id"]),
        "intent_id": str(case["intent_id"]),
        "seed": runtime.seed,
        "target_value_preserved": bool(case.get("target_value_preserved", False)),
        "coordinated_parameters": tuple(str(x) for x in case.get("coordinated_parameters", ())),
        "modification_distance": _modification_distance(case),
        "solver_seconds": float(case.get("solver_seconds", 0.0)),
        "peak_memory_mib": None,
        "campaign_test_index": campaign_test_index,
        "constraints_exercised": tuple(str(x) for x in case.get("violated_constraints", ())),
        "boundaries_exercised": (_boundary_label(case),),
        "guard_transitions": (),
        "topologies": ("single_npu_pilot",),
        "feature_interactions": (),
        "backend_paths": ("qwen2", "mindspeed_llm_26.1"),
        "solver_modifications": (
            dict(case.get("assignments", {}))
            if isinstance(case.get("assignments"), Mapping)
            else {}
        ),
    }

    if status != "ready":
        outcome = (
            ExperimentOutcome.EXPECTED_REJECTION
            if status in {"filtered", "unsat"}
            else ExperimentOutcome.UNKNOWN
        )
        elapsed = time.monotonic() - campaign_started
        return ExperimentRunRecord(
            **common,
            generated=False,
            deepest_milestone=ExecutionMilestone.CONFIG_GENERATION,
            outcome=outcome,
            duration_seconds=0.0,
            gpu_seconds=0.0,
            timed_out=False,
            campaign_elapsed_seconds=elapsed,
            campaign_gpu_seconds=cumulative_gpu_seconds,
            metadata={
                "pilot_only_not_final_metrics": True,
                "planner_status": status,
                "preflight": case.get("preflight"),
                "violated_constraints": list(case.get("violated_constraints", ())),
                "unknown_constraints": list(case.get("unknown_constraints", ())),
            },
        )

    assignments = case.get("assignments", {})
    if not isinstance(assignments, Mapping):
        raise ValueError(f"case {case['case_id']}: assignments must be an object")
    try:
        cli_args = assignments_to_qwen2_cli(assignments)
    except (KeyError, ValueError, TypeError) as exc:
        elapsed = time.monotonic() - campaign_started
        return ExperimentRunRecord(
            **common,
            generated=False,
            deepest_milestone=ExecutionMilestone.CONFIG_GENERATION,
            outcome=ExperimentOutcome.UNKNOWN,
            duration_seconds=0.0,
            gpu_seconds=0.0,
            timed_out=False,
            campaign_elapsed_seconds=elapsed,
            campaign_gpu_seconds=cumulative_gpu_seconds,
            metadata={
                "pilot_only_not_final_metrics": True,
                "planner_status": status,
                "unsupported_runtime_binding": str(exc),
                "assignments": dict(assignments),
            },
        )

    case_root = runtime.output_root / str(case["case_id"])
    env = os.environ.copy()
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": str(runtime.device),
            "OUT_ROOT": str(case_root),
            "MASTER_PORT": str(runtime.master_port + campaign_test_index),
            "TP": "1",
            "PP": "1",
            "NPROC": "1",
            "INITIAL_TRAIN_ITERS": "1",
            "CONTINUE_TRAIN_ITERS": "1",
            "SKIP_RELOAD": "1",
            "SAVE_CHECKPOINTS": "0",
        }
    )
    command = ["bash", str(runtime.launcher), *cli_args]
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=runtime.timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        output = completed.stdout or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        if isinstance(exc.stderr, str):
            output += exc.stderr
    duration = time.monotonic() - started
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "pilot-runner.log").write_text(output, encoding="utf-8", errors="replace")

    milestone = classify_milestone(output, returncode, timed_out)
    outcome = classify_outcome(output, returncode, timed_out)
    elapsed = time.monotonic() - campaign_started
    return ExperimentRunRecord(
        **common,
        generated=True,
        deepest_milestone=milestone,
        outcome=outcome,
        duration_seconds=duration,
        gpu_seconds=duration,
        timed_out=timed_out,
        campaign_elapsed_seconds=elapsed,
        campaign_gpu_seconds=cumulative_gpu_seconds + duration,
        metadata={
            "pilot_only_not_final_metrics": True,
            "planner_status": status,
            "preflight": case.get("preflight"),
            "assignments": dict(assignments),
            "command": command,
            "returncode": returncode,
            "device": runtime.device,
            "run_root": str(case_root),
            "native_validator_mode": (
                "embedded_framework_validation_pilot"
                if method is ExperimentMethod.NATIVE_VALIDATOR_GUIDED
                else None
            ),
        },
    )


def assignments_to_qwen2_cli(assignments: Mapping[str, Any]) -> list[str]:
    argv: list[str] = []
    for raw_name, value in sorted(assignments.items()):
        leaf = str(raw_name).rsplit(".", 1)[-1]
        if leaf in _VALUE_FLAGS:
            if isinstance(value, bool) or value is None:
                raise TypeError(f"{raw_name}: scalar value required")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{raw_name}: non-finite value")
            argv.extend((_VALUE_FLAGS[leaf], str(value)))
            continue
        if leaf in _TRUE_FLAGS:
            if value is True:
                argv.append(_TRUE_FLAGS[leaf])
                continue
            raise ValueError(f"{raw_name}: false boolean has no safe pilot inverse flag")
        raise KeyError(f"{raw_name}: no Qwen2 pilot CLI binding")
    return argv


def classify_milestone(output: str, returncode: int, timed_out: bool) -> ExecutionMilestone:
    if timed_out:
        return ExecutionMilestone.TIMEOUT
    lowered = output.lower()
    if returncode == 0 and ("iteration" in lowered or "successfully saved checkpoint" in lowered):
        return ExecutionMilestone.OPTIMIZER_STEP
    if "aclnntopk" in lowered or "forward_step" in lowered:
        return ExecutionMilestone.FORWARD
    if (
        "building gpt model" in lowered
        or "setting up optimizer" in lowered
        or "optimizer_param_scheduler" in lowered
        or "bias_swiglu" in lowered
    ):
        return ExecutionMilestone.MODEL_CONSTRUCTION
    if "initialized tensor model parallel" in lowered or "initialize distributed" in lowered:
        return ExecutionMilestone.PROCESS_GROUP_INITIALIZATION
    if "arguments" in lowered or returncode != 0:
        return ExecutionMilestone.CONFIG_VALIDATION
    return ExecutionMilestone.ARGUMENT_PARSING


def classify_outcome(output: str, returncode: int, timed_out: bool) -> ExperimentOutcome:
    if timed_out:
        return ExperimentOutcome.UNEXPLAINED_FAILURE
    if returncode == 0:
        return ExperimentOutcome.VALID
    lowered = output.lower()
    if any(pattern.lower() in lowered for pattern in _INFRA_PATTERNS):
        return ExperimentOutcome.INFRASTRUCTURE_FAILURE
    if any(pattern.lower() in lowered for pattern in _RESOURCE_PATTERNS):
        return ExperimentOutcome.RESOURCE_FAILURE
    if "traceback" in lowered or "assertionerror" in lowered or "valueerror" in lowered or "runtimeerror" in lowered:
        return ExperimentOutcome.EXPECTED_REJECTION
    return ExperimentOutcome.UNEXPLAINED_FAILURE


def dump_records(records: Sequence[ExperimentRunRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )


def _boundary_label(case: Mapping[str, Any]) -> str:
    target = str(case.get("target_parameter", "unknown"))
    return f"{target}={case.get('target_value')!r}"


def _modification_distance(case: Mapping[str, Any]) -> float | None:
    assignments = case.get("assignments")
    if not isinstance(assignments, Mapping):
        return None
    target = case.get("target_parameter")
    metadata = case.get("metadata")
    baseline_value = None
    if isinstance(metadata, Mapping):
        baseline_value = metadata.get("baseline_value")
    if baseline_value is None or target not in assignments:
        return None
    value = assignments[target]
    if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(baseline_value, (int, float)) and not isinstance(baseline_value, bool):
        return abs(float(value) - float(baseline_value))
    return 0.0 if value == baseline_value else 1.0
