#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.rq1_secondary_review import compare_primary_secondary_reviews


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare independent RQ1 primary and secondary adjudications."
    )
    parser.add_argument("--primary-adjudication", type=Path, required=True)
    parser.add_argument("--secondary-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = compare_primary_secondary_reviews(
        args.primary_adjudication,
        args.secondary_review,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": payload["complete"],
                "adjudication_required_count": payload["adjudication_required_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
