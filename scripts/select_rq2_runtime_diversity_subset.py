#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


def _read_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("RQ2 plan must contain a cases array")
    return value


def _intent_hash(intent_id: str) -> str:
    return hashlib.sha256(intent_id.encode("utf-8")).hexdigest()


def select_subset(plan: Mapping[str, Any], fraction: float = 0.2) -> dict[str, Any]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    cases = [item for item in plan.get("cases", ()) if isinstance(item, Mapping)]
    by_workload: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        by_workload[str(case["workload_id"])].add(str(case["intent_id"]))

    selected: set[str] = set()
    workload_counts: dict[str, dict[str, int]] = {}
    for workload_id, intent_ids in sorted(by_workload.items()):
        ordered = sorted(intent_ids, key=lambda item: (_intent_hash(item), item))
        count = max(1, int(len(ordered) * fraction))
        chosen = ordered[:count]
        selected.update(chosen)
        workload_counts[workload_id] = {
            "available_intents": len(ordered),
            "selected_intents": len(chosen),
        }

    subset_cases = [copy.deepcopy(dict(case)) for case in cases if str(case["intent_id"]) in selected]
    payload = copy.deepcopy(dict(plan))
    payload["cases"] = subset_cases
    payload["case_count"] = len(subset_cases)
    payload["intent_count"] = len(selected)
    payload["method_counts"] = dict(sorted(Counter(str(case["method"]) for case in subset_cases).items()))
    payload["status_counts"] = dict(sorted(Counter(str(case.get("status", "unknown")) for case in subset_cases).items()))
    metadata = payload.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    metadata["runtime_diversity_subset"] = {
        "fraction": fraction,
        "selection": "lowest SHA-256(intent_id) within each workload",
        "workloads": workload_counts,
    }
    payload["metadata"] = metadata
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the fixed hash-based RQ2 runtime-diversity intent subset.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    args = parser.parse_args()

    payload = select_subset(_read_plan(args.plan), args.fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "intent_count": payload["intent_count"],
                "case_count": payload["case_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
