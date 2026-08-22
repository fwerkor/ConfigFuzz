#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from configfuzz.experiment import ExperimentMethod
from configfuzz.experiment_campaign import load_campaign_workloads
from configfuzz.rq2_gpu_executor import (
    RUNTIME_INSTRUMENTATION_VERSION,
    _available_master_port,
    execute_campaign_case,
    _file_sha256,
    _git_revision,
    load_plan,
)


def _run_id(framework_id: str, case: Mapping[str, Any], seed: int) -> str:
    material = f"rq3-current:{framework_id}:{case['case_id']}:{seed}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"rq3-current-{framework_id}-{digest}"


def _cleanup_checkpoint_tree(case_root: Path) -> bool:
    checkpoint_root = case_root / "checkpoints"
    try:
        if checkpoint_root.is_symlink() or checkpoint_root.is_file():
            checkpoint_root.unlink()
        elif checkpoint_root.is_dir():
            shutil.rmtree(checkpoint_root)
        else:
            return False
    except OSError as exc:
        print(
            json.dumps(
                {
                    "event": "rq3_checkpoint_cleanup_failed",
                    "path": str(checkpoint_root),
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def _load_resume_state(path: Path) -> tuple[set[str], dict[ExperimentMethod, int], dict[ExperimentMethod, float], float]:
    completed: set[str] = set()
    generated: dict[ExperimentMethod, int] = defaultdict(int)
    accelerator_seconds: dict[ExperimentMethod, float] = defaultdict(float)
    total_accelerator_seconds = 0.0
    if not path.is_file():
        return completed, generated, accelerator_seconds, total_accelerator_seconds
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            continue
        run_id = payload.get("run_id")
        if run_id:
            completed.add(str(run_id))
        try:
            method = ExperimentMethod(str(payload["method"]))
        except (KeyError, ValueError):
            continue
        if bool(payload.get("generated", False)):
            generated[method] += 1
        seconds = float(payload.get("accelerator_seconds", payload.get("gpu_seconds", 0.0)) or 0.0)
        accelerator_seconds[method] += seconds
        total_accelerator_seconds += seconds
    return completed, generated, accelerator_seconds, total_accelerator_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable RQ3 current-version GPU discovery under frozen per-method budgets.")
    parser.add_argument("--framework", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--device-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--master-port", type=int, default=32001)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-generated-tests-per-method", type=int, default=2000)
    parser.add_argument("--max-accelerator-hours-per-method", type=float, default=24.0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.max_generated_tests_per_method <= 0:
        raise ValueError("max generated tests per method must be positive")
    if args.max_accelerator_hours_per_method <= 0:
        raise ValueError("max accelerator hours per method must be positive")

    plan = load_plan(args.plan)
    workloads = load_campaign_workloads(args.workloads)
    launcher = args.launcher.resolve()
    output_root = args.output_root.resolve()
    output = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    completed, generated_by_method, seconds_by_method, cumulative_seconds = _load_resume_state(output)
    for run_id in completed:
        _cleanup_checkpoint_tree(output_root / run_id)
    max_seconds = args.max_accelerator_hours_per_method * 3600.0
    provenance = {
        "runner_revision": _git_revision(),
        "plan_sha256": _file_sha256(args.plan),
        "workload_registry_sha256": _file_sha256(args.workloads),
        "launcher_sha256": _file_sha256(launcher),
        "runtime_instrumentation": RUNTIME_INSTRUMENTATION_VERSION,
        "rq3_phase": "current_version_discovery",
    }

    campaign_started = time.monotonic()
    written = 0
    launched = 0
    skipped_existing = 0
    skipped_budget = 0
    candidate_failures = 0
    stop_reasons: dict[str, str] = {}

    with output.open("a", encoding="utf-8") as handle:
        for index, case in enumerate(plan["cases"], 1):
            if not isinstance(case, Mapping):
                continue
            method = ExperimentMethod(str(case["method"]))
            if generated_by_method[method] >= args.max_generated_tests_per_method:
                stop_reasons[method.value] = "generated_test_budget"
                skipped_budget += 1
                continue
            if seconds_by_method[method] >= max_seconds:
                stop_reasons[method.value] = "accelerator_hour_budget"
                skipped_budget += 1
                continue

            run_id = _run_id(args.framework, case, args.seed)
            if run_id in completed:
                skipped_existing += 1
                continue
            workload_id = str(case["workload_id"])
            workload = workloads.get(workload_id)
            if workload is None:
                raise KeyError(f"plan references workload not supported by {args.framework}: {workload_id}")

            # A prior interrupted attempt may have left a partial checkpoint tree.
            # Remove it before retrying so every case starts from the same clean state.
            _cleanup_checkpoint_tree(output_root / run_id)
            record, used_accelerator = execute_campaign_case(
                framework_id=args.framework,
                case=case,
                workload=workload,
                launcher=launcher,
                output_root=output_root,
                gpu_devices=args.gpus,
                device_count=args.device_count,
                accelerator_kind="gpu",
                seed=args.seed,
                master_port=_available_master_port(args.master_port + (index % 300)),
                timeout_seconds=args.timeout_seconds,
                run_id=run_id,
                campaign_test_index=index,
                campaign_started=campaign_started,
                cumulative_accelerator_seconds=cumulative_seconds,
                provenance=provenance,
                rq="rq3",
                skip_checkpoint=False,
            )
            cumulative_seconds += record.gpu_seconds
            if record.generated:
                generated_by_method[method] += 1
            seconds_by_method[method] += record.gpu_seconds

            payload = record.to_dict()
            payload["campaign_gpu_seconds"] = cumulative_seconds
            payload["campaign_accelerator_seconds"] = cumulative_seconds
            metadata = dict(payload.get("metadata", {}))
            metadata.update(
                {
                    "rq3_phase": "current_version_discovery",
                    "triage_state": "pending" if record.outcome.value == "unexplained_failure" else "not_candidate",
                    "budget_max_generated_tests_per_method": args.max_generated_tests_per_method,
                    "budget_max_accelerator_hours_per_method": args.max_accelerator_hours_per_method,
                    "checkpoint_path_enabled": True,
                    "checkpoint_retention": "ephemeral",
                    "checkpoint_cleanup_after_result_commit": True,
                }
            )
            payload["metadata"] = metadata
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            # Checkpoint save/load is part of the RQ3 execution semantics, but the
            # resulting files are not evidence once the durable result is committed.
            _cleanup_checkpoint_tree(output_root / run_id)
            completed.add(run_id)
            written += 1
            launched += int(used_accelerator)

            if record.outcome.value == "unexplained_failure":
                candidate_failures += 1
                print(
                    json.dumps(
                        {
                            "event": "rq3_candidate_failure",
                            "framework": args.framework,
                            "method": method.value,
                            "run_id": run_id,
                            "workload": workload_id,
                            "milestone": record.deepest_milestone.value,
                            "log": payload.get("metadata", {}).get("log_path"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if written % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "rq3_progress",
                            "framework": args.framework,
                            "written": written,
                            "launched": launched,
                            "candidate_failures": candidate_failures,
                            "generated_by_method": {m.value: generated_by_method[m] for m in ExperimentMethod},
                            "accelerator_hours_by_method": {
                                m.value: round(seconds_by_method[m] / 3600.0, 6) for m in ExperimentMethod
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    summary = {
        "framework": args.framework,
        "written": written,
        "launched": launched,
        "skipped_existing": skipped_existing,
        "skipped_budget": skipped_budget,
        "candidate_failures": candidate_failures,
        "generated_by_method": {m.value: generated_by_method[m] for m in ExperimentMethod},
        "accelerator_hours_by_method": {m.value: seconds_by_method[m] / 3600.0 for m in ExperimentMethod},
        "stop_reasons": stop_reasons,
        "output": str(output),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
