#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.experiment import ExperimentMethod, write_json
from configfuzz.experiment_campaign import (
    load_campaign_workloads,
    load_frozen_intents,
    plan_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expand frozen RQ2 intents into identical comparison-method cases."
    )
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workload",
        action="append",
        help="plan only these workload IDs after verifying the frozen intent hash",
    )
    parser.add_argument(
        "--intent",
        action="append",
        help="plan only these intent IDs after verifying the frozen intent hash",
    )
    parser.add_argument(
        "--solver-timeout-ms",
        type=int,
        default=1000,
        help="per-case Z3 timeout for solver-based methods (default: 1000)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        help="runtime distributed world size supplied as immutable solver context",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=tuple(item.value for item in ExperimentMethod),
        help="method to include; repeatable (defaults to all RQ2 methods)",
    )
    args = parser.parse_args()

    methods = (
        tuple(ExperimentMethod(item) for item in args.method)
        if args.method
        else (
            ExperimentMethod.RAW_MUTATION,
            ExperimentMethod.NATIVE_VALIDATOR_GUIDED,
            ExperimentMethod.CONSTRAINT_FILTER_ONLY,
            ExperimentMethod.STATIC_HARD_CONFIGFUZZ,
            ExperimentMethod.CONFIGFUZZ,
            ExperimentMethod.GLOBAL_REPAIR,
        )
    )
    workloads = load_campaign_workloads(args.workloads)
    intents = load_frozen_intents(args.intents)
    if args.workload:
        selected_workloads = set(args.workload)
        intents = [item for item in intents if item.workload_id in selected_workloads]
    if args.intent:
        selected_intents = set(args.intent)
        intents = [item for item in intents if item.intent_id in selected_intents]
    if not intents:
        parser.error("intent filters selected no frozen intents")
    payload = plan_campaign(
        workloads,
        intents,
        methods=methods,
        solver_timeout_ms=args.solver_timeout_ms,
        runtime_context=(
            {
                "world_size": args.world_size,
                "args": {"world_size": args.world_size},
            }
            if args.world_size is not None
            else None
        ),
    )
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "intent_count": payload["intent_count"],
                "case_count": payload["case_count"],
                "method_counts": payload["method_counts"],
                "status_counts": payload["status_counts"],
                "solver_timeout_ms": payload["solver_timeout_ms"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
