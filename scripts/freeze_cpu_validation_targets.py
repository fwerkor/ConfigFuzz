#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.cpu_campaign import build_frozen_cpu_targets, dump_frozen_cpu_targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze deterministic CPU validation interventions.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-subject", type=int, default=6)
    parser.add_argument("--solver-timeout-ms", type=int, default=3000)
    args = parser.parse_args()
    payload = build_frozen_cpu_targets(
        args.root.resolve(),
        limit_per_subject=args.limit_per_subject,
        solver_timeout_ms=args.solver_timeout_ms,
    )
    dump_frozen_cpu_targets(payload, args.output)
    print(json.dumps(payload["frozen"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
