#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from rq1_pairs_26_1 import CASES, ROOT


PERSISTENT_LAUNCHER = ROOT / "experiments/runtime/mindspeed_26_1_persistent.sh"
DEFAULT_RUN_ROOT = Path("/model/cyh/configfuzz-runs/rq1-persistent-26.1")
DEFAULT_REFERENCE = ROOT / "experiments/results/rq1.26.1.single-npu.jsonl"


def _member_id(constraint_id: str, role: str) -> str:
    return f"{constraint_id}__{role}"


def _load_reference(path: Path, selected_ids: set[str]) -> tuple[int, float]:
    if not path.exists():
        return 0, 0.0
    count = 0
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = _member_id(str(record.get("constraint_id")), str(record.get("pair_role")))
        if key not in selected_ids:
            continue
        count += 1
        total += float(record.get("duration_seconds", 0.0))
    return count, total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B benchmark RQ1 cold-per-case execution against same-topology persistent MindSpeed workers."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=6200)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--constraint", action="append", dest="constraints")
    parser.add_argument("--timeout-per-case", type=float, default=180.0)
    args = parser.parse_args()

    selected = [c for c in CASES if not args.constraints or c.constraint_id in set(args.constraints)]
    if not selected:
        raise SystemExit("no matching RQ1 cases")

    grouped = defaultdict(list)
    selected_member_ids: set[str] = set()
    for case in selected:
        for role in ("satisfying", "violating"):
            overrides = case.satisfying_args if role == "satisfying" else case.violating_args
            train_iters = case.satisfying_train_iters if role == "satisfying" else case.violating_train_iters
            member_id = _member_id(case.constraint_id, role)
            selected_member_ids.add(member_id)
            grouped[case.profile].append(
                {
                    "case_id": member_id,
                    "args": list(overrides),
                    "train_iters": train_iters,
                }
            )

    args.run_root.mkdir(parents=True, exist_ok=True)
    group_summaries = []
    total_wall = 0.0
    total_hot = 0.0
    total_worker_warm = 0.0
    completed = 0

    for offset, (profile, members) in enumerate(sorted(grouped.items())):
        group_root = args.run_root / profile
        group_root.mkdir(parents=True, exist_ok=True)
        manifest = group_root / "manifest.json"
        worker_output = group_root / "worker.jsonl"
        worker_stdout = group_root / "worker.stdout.log"
        case_logs = group_root / "cases"
        manifest.write_text(json.dumps({"cases": members}, indent=2) + "\n", encoding="utf-8")

        env = os.environ.copy()
        env.update(
            {
                "PROFILE": profile,
                "MANIFEST": str(manifest),
                "OUTPUT": str(worker_output),
                "CASE_LOG_ROOT": str(case_logs),
                "ASCEND_RT_VISIBLE_DEVICES": str(args.device),
                "NPROC": "1",
                "TP": "1",
                "PP": "1",
                "MASTER_PORT": str(args.base_port + offset),
            }
        )
        started = time.monotonic()
        completed_proc = subprocess.run(
            ["bash", str(PERSISTENT_LAUNCHER)],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(300.0, args.timeout_per_case * len(members) + 180.0),
            check=False,
        )
        wall = time.monotonic() - started
        total_wall += wall
        worker_stdout.write_text(completed_proc.stdout or "", encoding="utf-8", errors="replace")

        records = []
        if worker_output.exists():
            records = [
                json.loads(line)
                for line in worker_output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        completed += len(records)
        hot = sum(float(item.get("duration_seconds", 0.0)) for item in records)
        warm = float(records[0].get("worker_warm_seconds", 0.0)) if records else 0.0
        total_hot += hot
        total_worker_warm += warm
        group_summaries.append(
            {
                "profile": profile,
                "requested": len(members),
                "completed": len(records),
                "returncode": completed_proc.returncode,
                "wall_seconds": wall,
                "worker_warm_seconds": warm,
                "hot_case_seconds": hot,
                "mean_hot_case_seconds": hot / len(records) if records else None,
                "worker_output": str(worker_output),
                "stdout_log": str(worker_stdout),
            }
        )

    ref_count, ref_total = _load_reference(args.reference, selected_member_ids)
    summary = {
        "requested_cases": len(selected_member_ids),
        "completed_cases": completed,
        "persistent_wall_seconds": total_wall,
        "persistent_worker_warm_seconds": total_worker_warm,
        "persistent_hot_case_seconds": total_hot,
        "persistent_mean_hot_case_seconds": total_hot / completed if completed else None,
        "cold_reference_count": ref_count,
        "cold_reference_total_seconds": ref_total,
        "cold_reference_mean_seconds": ref_total / ref_count if ref_count else None,
        "wall_speedup_vs_reference": ref_total / total_wall if ref_total and total_wall else None,
        "groups": group_summaries,
    }
    summary_path = args.run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"WROTE {summary_path}")
    return 0 if completed == len(selected_member_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
