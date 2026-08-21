#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.experiment import ExperimentMethod
from configfuzz.rq2_gpu_executor import execute_primary_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a resumable formal RQ2 GPU campaign on a promoted framework subject.")
    parser.add_argument("--framework", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--device-count", type=int, default=2)
    parser.add_argument("--accelerator-kind", default="gpu")
    parser.add_argument("--harness-path", action="append", default=[])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--master-port", type=int, default=30001)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--workload", action="append")
    parser.add_argument("--intent", action="append")
    parser.add_argument("--method", action="append", choices=tuple(item.value for item in ExperimentMethod))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    methods = tuple(ExperimentMethod(item) for item in (args.method or ()))
    summary = execute_primary_campaign(
        framework_id=args.framework,
        plan_path=args.plan,
        workload_registry_path=args.workloads,
        launcher=args.launcher,
        output_root=args.output_root,
        output_jsonl=args.output,
        gpu_devices=args.gpus,
        device_count=args.device_count,
        accelerator_kind=args.accelerator_kind,
        harness_paths=tuple(args.harness_path),
        seed=args.seed,
        master_port=args.master_port,
        timeout_seconds=args.timeout_seconds,
        workload_ids=tuple(args.workload or ()),
        methods=methods,
        intent_ids=tuple(args.intent or ()),
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
