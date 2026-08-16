#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CANDIDATE_ID = "mindspeed-03bb94e94855"
BUGGY_COMMIT = "708af43cf9d42ec729a5cbb65bf20e0106bb67bd"
FIXED_COMMIT = "03bb94e948552071b5abe66a0ec6f04f91404ed9"
EXPECTED_ERROR = "recompute_num_layers must be None or 0 when using recompute_in_bubble"


def run_member() -> int:
    from argparse import Namespace
    from mindspeed.features_manager.pipeline_parallel.ripipe_schedules_feature import (
        RiPipeSchedulesBubbleFeature,
    )

    args = Namespace(
        adaptive_recompute_device_size=None,
        adaptive_recompute_device_swap=False,
        recompute_in_advance=False,
        recompute_in_bubble=True,
        optimize_send_recv_comm=False,
        ampipe_degree=1,
        adaptive_memory_optimization=False,
        recompute_num_layers=None,
        pipeline_model_parallel_size=2,
        num_layers_per_virtual_pipeline_stage=2,
        num_layers=8,
        enable_recompute_layers_per_pp_rank=False,
        swap_attention=False,
        recompute_granularity=None,
        recompute_method=None,
    )
    feature = RiPipeSchedulesBubbleFeature()
    feature.validate_args(args)
    print(
        "FIRST",
        args.recompute_num_layers,
        args.recompute_granularity,
        args.recompute_method,
        flush=True,
    )
    try:
        feature.validate_args(args)
    except Exception as exc:  # historical oracle is the exact second-pass assertion
        print(f"SECOND_EXCEPTION {type(exc).__name__}: {exc}", flush=True)
        return 42 if isinstance(exc, AssertionError) and EXPECTED_ERROR in str(exc) else 43
    print(
        "SECOND",
        args.recompute_num_layers,
        args.recompute_granularity,
        args.recompute_method,
        flush=True,
    )
    if (
        args.recompute_num_layers == 2
        and args.recompute_granularity == "full"
        and args.recompute_method == "block"
    ):
        return 0
    return 44


def git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def execute(root: Path, revision: str, repetition: int) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, __file__, "--member"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    duration = time.monotonic() - started
    output = completed.stdout or ""
    if revision == "buggy":
        oracle_matched = completed.returncode == 42 and EXPECTED_ERROR in output
    else:
        oracle_matched = completed.returncode == 0 and "SECOND 2 full block" in output
    return {
        "candidate_id": CANDIDATE_ID,
        "repository": "MindSpeed",
        "revision": revision,
        "commit": git_head(root),
        "repetition": repetition,
        "returncode": completed.returncode,
        "duration_seconds": duration,
        "oracle_matched": oracle_matched,
        "oracle": (
            "second identical validate_args call raises the historical assertion"
            if revision == "buggy"
            else "second identical validate_args call remains idempotent"
        ),
        "trigger": {
            "recompute_in_bubble": True,
            "pipeline_model_parallel_size": 2,
            "num_layers": 8,
            "num_layers_per_virtual_pipeline_stage": 2,
            "enable_recompute_layers_per_pp_rank": False,
        },
        "observed_output": output.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="store_true")
    parser.add_argument("--buggy-root", type=Path)
    parser.add_argument("--fixed-root", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.member:
        return run_member()
    if not args.buggy_root or not args.fixed_root or not args.output:
        parser.error("--buggy-root, --fixed-root, and --output are required")
    if git_head(args.buggy_root) != BUGGY_COMMIT:
        raise SystemExit("buggy worktree commit mismatch")
    if git_head(args.fixed_root) != FIXED_COMMIT:
        raise SystemExit("fixed worktree commit mismatch")

    records: list[dict[str, object]] = []
    for revision, root in (("buggy", args.buggy_root), ("fixed", args.fixed_root)):
        for repetition in range(1, args.repetitions + 1):
            record = execute(root, revision, repetition)
            records.append(record)
            print(
                f"{revision} rep={repetition} rc={record['returncode']} "
                f"oracle={record['oracle_matched']} duration={record['duration_seconds']:.3f}s",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    buggy_ok = all(r["oracle_matched"] for r in records if r["revision"] == "buggy")
    fixed_ok = all(r["oracle_matched"] for r in records if r["revision"] == "fixed")
    summary = {
        "candidate_id": CANDIDATE_ID,
        "buggy_commit": BUGGY_COMMIT,
        "fixed_commit": FIXED_COMMIT,
        "buggy_reproduced_three_times": buggy_ok and args.repetitions >= 3,
        "fixed_passed": fixed_ok,
        "root_cause_matched": buggy_ok and fixed_ok,
        "minimum_configuration_recorded": True,
        "repetitions_per_revision": args.repetitions,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["buggy_reproduced_three_times"] and fixed_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
