#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from configfuzz.experiment import freeze_intent_file
from configfuzz.intent_selection import select_balanced_intents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a balanced fixed-size RQ2 intent set from a larger candidate pool."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-output", type=Path)
    parser.add_argument("--default-per-primary-workload", type=int, default=300)
    args = parser.parse_args()

    payload = select_balanced_intents(
        args.candidates,
        args.workloads,
        default_per_primary_workload=args.default_per_primary_workload,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    result: dict[str, object] = {
        "output": str(args.output),
        "intent_count": payload["metadata"]["intent_count"],
        "workload_selection": payload["metadata"]["workload_selection"],
    }
    if args.frozen_output is not None:
        frozen = freeze_intent_file(args.output, args.frozen_output)
        result["frozen_output"] = str(args.frozen_output)
        result["sha256"] = frozen["frozen"]["sha256"]
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
