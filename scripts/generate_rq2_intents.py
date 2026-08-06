#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from configfuzz.experiment import freeze_intent_file
from configfuzz.intent_generation import generate_intent_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and optionally freeze an RQ2 mutation-intent set."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-output", type=Path)
    parser.add_argument(
        "--skip-unbound",
        action="store_true",
        help="skip workload entries whose baseline_config is not yet bound",
    )
    parser.add_argument(
        "--topology-value",
        action="append",
        type=int,
        dest="topology_values",
        help="parallel size to include; repeatable (defaults: 1, 2, 4, 8)",
    )
    args = parser.parse_args()

    payload = generate_intent_payload(
        args.corpus,
        args.workloads,
        skip_unbound=args.skip_unbound,
        topology_values=tuple(args.topology_values or (1, 2, 4, 8)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    result = {
        "output": str(args.output),
        "workload_count": payload["metadata"]["workload_count"],
        "intent_count": payload["metadata"]["intent_count"],
        "intents_per_workload": payload["metadata"]["intents_per_workload"],
    }
    if args.frozen_output is not None:
        frozen = freeze_intent_file(args.output, args.frozen_output)
        result["frozen_output"] = str(args.frozen_output)
        result["sha256"] = frozen["frozen"]["sha256"]
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
