#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configfuzz.rq2_gpu_executor import execute_primary_campaign
from scripts.assemble_rq2_selective_reuse import assemble


def run_reuse_campaign(
    *,
    framework: str,
    plan: Path,
    workloads: Path,
    launcher: Path,
    output: Path,
    runtime_root: Path,
    assembly_root: Path,
    accelerator_kind: str,
    devices: str,
    device_count: int,
    seed: int,
    master_port: int,
    timeout_seconds: float,
    source_results: Sequence[Path],
    harness_paths: Sequence[str | Path],
    max_infra_retries: int,
) -> dict[str, object]:
    runtime_root.mkdir(parents=True, exist_ok=True)
    assembly_root.mkdir(parents=True, exist_ok=True)
    sources = list(source_results)
    attempts: list[dict[str, object]] = []

    # Attempt 0 is a planning-only assembly pass.  With no reusable source it
    # collapses all ready logical cases to one representative per byte-equivalent
    # materialized runtime configuration before any accelerator is launched.
    for attempt in range(max_infra_retries + 1):
        missing_plan = assembly_root / f"runtime-plan-attempt-{attempt}.json"
        manifest = assembly_root / f"assembly-attempt-{attempt}.json"
        summary = assemble(
            framework=framework,
            plan_path=plan,
            workload_registry=workloads,
            launcher=launcher,
            source_paths=sources,
            output=output,
            missing_plan=missing_plan,
            manifest=manifest,
            seed=seed,
            harness_paths=harness_paths,
            accelerator_kind=accelerator_kind,
        )
        attempts.append(
            {
                "attempt": attempt,
                "assembly": summary,
                "runtime_output": None,
            }
        )
        if bool(summary["complete"]):
            return {
                "framework_id": framework,
                "complete": True,
                "accelerator_kind": accelerator_kind,
                "attempts": attempts,
                "output": str(output),
                "planned_cases": summary["planned_cases"],
                "physical_runtime_configurations": sum(
                    int(item["assembly"]["unique_missing_runtime_configurations"])
                    for item in attempts[:-1]
                ),
            }

        runtime_output = runtime_root / f"runtime-attempt-{attempt}.jsonl"
        run_summary = execute_primary_campaign(
            framework_id=framework,
            plan_path=missing_plan,
            workload_registry_path=workloads,
            launcher=launcher,
            output_root=runtime_root / f"cases-attempt-{attempt}",
            output_jsonl=runtime_output,
            gpu_devices=devices,
            device_count=device_count,
            accelerator_kind=accelerator_kind,
            harness_paths=harness_paths,
            seed=seed,
            master_port=master_port + attempt * 500,
            timeout_seconds=timeout_seconds,
        )
        sources.append(runtime_output)
        attempts[-1]["runtime_output"] = str(runtime_output)
        attempts[-1]["runtime"] = run_summary

    final_missing = assembly_root / "runtime-plan-final-missing.json"
    final_manifest = assembly_root / "assembly-final.json"
    final_summary = assemble(
        framework=framework,
        plan_path=plan,
        workload_registry=workloads,
        launcher=launcher,
        source_paths=sources,
        output=output,
        missing_plan=final_missing,
        manifest=final_manifest,
        seed=seed,
        harness_paths=harness_paths,
        accelerator_kind=accelerator_kind,
    )
    return {
        "framework_id": framework,
        "complete": bool(final_summary["complete"]),
        "accelerator_kind": accelerator_kind,
        "attempts": attempts,
        "final_assembly": final_summary,
        "output": str(output) if bool(final_summary["complete"]) else None,
        "planned_cases": final_summary["planned_cases"],
        "physical_runtime_configurations": sum(
            int(item["assembly"]["unique_missing_runtime_configurations"])
            for item in attempts
        ),
        "remaining_unique_runtime_configurations": final_summary[
            "unique_missing_runtime_configurations"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an RQ2 campaign with pre-execution runtime-equivalence deduplication, "
            "provenance-checked reuse, and expansion back to all logical method/intent records."
        )
    )
    parser.add_argument("--framework", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--assembly-root", type=Path, required=True)
    parser.add_argument("--accelerator-kind", default="gpu")
    parser.add_argument("--devices", default="4,5")
    parser.add_argument("--device-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--master-port", type=int, default=30001)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--source-result", action="append", type=Path, default=[])
    parser.add_argument("--harness-path", action="append", default=[])
    parser.add_argument("--max-infra-retries", type=int, default=2)
    args = parser.parse_args()
    if args.max_infra_retries < 0:
        parser.error("--max-infra-retries must be non-negative")
    if not args.harness_path:
        parser.error("at least one --harness-path is required for formal runtime reuse")

    payload = run_reuse_campaign(
        framework=args.framework,
        plan=args.plan,
        workloads=args.workloads,
        launcher=args.launcher.resolve(),
        output=args.output,
        runtime_root=args.runtime_root,
        assembly_root=args.assembly_root,
        accelerator_kind=args.accelerator_kind,
        devices=args.devices,
        device_count=args.device_count,
        seed=args.seed,
        master_port=args.master_port,
        timeout_seconds=args.timeout_seconds,
        source_results=args.source_result,
        harness_paths=args.harness_path,
        max_infra_retries=args.max_infra_retries,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if bool(payload["complete"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
