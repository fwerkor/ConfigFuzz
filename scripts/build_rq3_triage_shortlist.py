#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.bug_benchmark import (
    build_triage_shortlist,
    dump_triage_shortlist,
    load_fix_candidates,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a diverse manual-triage shortlist from historical fix candidates."
    )
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-per-primary-parameter", type=int, default=5)
    parser.add_argument("--max-patch-lines", type=int, default=2000)
    args = parser.parse_args()

    payload = build_triage_shortlist(
        load_fix_candidates(args.candidates),
        limit=args.limit,
        max_per_primary_parameter=args.max_per_primary_parameter,
        max_patch_lines=args.max_patch_lines,
    )
    dump_triage_shortlist(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shortlist_count": payload["shortlist_count"],
                "repository_counts": payload["repository_counts"],
                "eligible_candidate_count": payload["selection_policy"][
                    "eligible_candidate_count"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
