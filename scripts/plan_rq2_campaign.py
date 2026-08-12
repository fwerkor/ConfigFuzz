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
    payload = plan_campaign(workloads, intents, methods=methods)
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "intent_count": payload["intent_count"],
                "case_count": payload["case_count"],
                "method_counts": payload["method_counts"],
                "status_counts": payload["status_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
