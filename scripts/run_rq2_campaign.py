#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.experiment import ExperimentMethod
from configfuzz.rq2_executor import (
    Qwen2PilotRuntime,
    dump_records,
    execute_qwen2_pilot_cases,
    load_campaign_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute selected RQ2 campaign-plan cases on the qualified Qwen2 26.1 pilot runtime."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=6300)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--intent", action="append", default=[])
    parser.add_argument(
        "--method",
        action="append",
        choices=tuple(item.value for item in ExperimentMethod),
        default=[],
    )
    args = parser.parse_args()

    runtime = Qwen2PilotRuntime(
        launcher=args.launcher.resolve(),
        output_root=args.output_root.resolve(),
        device=args.device,
        master_port=args.master_port,
        timeout_seconds=args.timeout_seconds,
    )
    records = execute_qwen2_pilot_cases(
        load_campaign_plan(args.plan),
        runtime,
        intent_ids=args.intent,
        methods=tuple(ExperimentMethod(item) for item in args.method),
    )
    dump_records(records, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "record_count": len(records),
                "generated_count": sum(item.generated for item in records),
                "deep_count": sum(item.deepest_milestone.value == "optimizer_step" for item in records),
                "outcomes": {
                    value: sum(item.outcome.value == value for item in records)
                    for value in sorted({item.outcome.value for item in records})
                },
                "pilot_only_not_final_metrics": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
