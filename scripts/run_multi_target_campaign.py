#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.multi_target_campaign import execute_multi_target_campaign


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Continuously fuzz a framework with joint multi-target ConfigFuzz mutations."
    )
    parser.add_argument("--framework", required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "-r", "--rounds", type=int, required=True, help="number of mutation/test rounds"
    )
    parser.add_argument(
        "--mutnm",
        "--targets-per-mutation",
        dest="targets_per_mutation",
        type=int,
        default=2,
        help="number of distinct target parameters mutated together in each round",
    )
    parser.add_argument("--seed", type=int, required=True, help="campaign random seed")
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--device-count", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=30001)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--solver-timeout-ms", type=int, default=1000)
    parser.add_argument("--workload", action="append")
    parser.add_argument("--intent-pool", default="method_independent")
    args = parser.parse_args()

    summary = execute_multi_target_campaign(
        framework_id=args.framework,
        workload_registry_path=args.workloads,
        candidate_path=args.candidates,
        launcher=args.launcher,
        output_root=args.output_root,
        output_jsonl=args.output,
        rounds=args.rounds,
        targets_per_mutation=args.targets_per_mutation,
        seed=args.seed,
        gpu_devices=args.gpus,
        device_count=args.device_count,
        master_port=args.master_port,
        timeout_seconds=args.timeout_seconds,
        solver_timeout_ms=args.solver_timeout_ms,
        workload_ids=tuple(args.workload or ()),
        intent_pool=args.intent_pool,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
