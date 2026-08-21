#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.cpu_campaign import load_frozen_cpu_targets, run_frozen_cpu_subject


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one subject from the frozen CPU validation target set.")
    parser.add_argument("targets", type=Path)
    parser.add_argument("subject")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = load_frozen_cpu_targets(args.targets)
    result = run_frozen_cpu_subject(args.root.resolve(), frozen, args.subject)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
