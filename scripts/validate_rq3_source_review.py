#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.bug_benchmark import validate_source_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the RQ3 source review and buggy/fixed execution queue."
    )
    parser.add_argument("review", type=Path)
    parser.add_argument("--execution-queue", type=Path)
    args = parser.parse_args()

    result = validate_source_review(args.review, args.execution_queue)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
