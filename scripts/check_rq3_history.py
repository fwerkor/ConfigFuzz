#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def _git_ok(root: Path, *args: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all RQ3 historical commits and reusable harness blobs before accelerator replay.")
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--mindspeed-root", type=Path, required=True)
    parser.add_argument("--mindspeed-llm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.bindings.read_text(encoding="utf-8"))
    roots = {"MindSpeed": args.mindspeed_root.resolve(), "MindSpeed-LLM": args.mindspeed_llm_root.resolve()}
    records = []
    failed = 0
    for case in payload.get("cases", []):
        root = roots[str(case["repository"])]
        buggy = str(case["buggy_commit"])
        fixed = str(case["fix_commit"])
        commit_ok = _git_ok(root, "cat-file", "-e", buggy + "^{commit}") and _git_ok(root, "cat-file", "-e", fixed + "^{commit}")
        harness = case.get("source_harness")
        harness_status = "generic_search_workload"
        if isinstance(harness, dict) and harness.get("path"):
            path = str(harness["path"])
            fixed_blob = _git_ok(root, "cat-file", "-e", f"{fixed}:{path}")
            buggy_blob = _git_ok(root, "cat-file", "-e", f"{buggy}:{path}")
            mode = str(harness.get("mode", ""))
            harness_ok = fixed_blob and (buggy_blob or mode == "copy_fixed_test_to_buggy_checkout")
            harness_status = "verified" if harness_ok else "missing_blob"
        else:
            harness_ok = True
        ok = commit_ok and harness_ok
        failed += not ok
        records.append(
            {
                "logical_bug_id": case["logical_bug_id"],
                "repository": case["repository"],
                "commit_pair_available": commit_ok,
                "harness_status": harness_status,
                "generic_workload": case["workload_id"],
                "ready": ok,
            }
        )
    report = {
        "schema_version": 1,
        "name": "rq3-historical-source-preflight",
        "logical_case_count": len(records),
        "ready_count": len(records) - failed,
        "failed_count": failed,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"logical_case_count": len(records), "ready_count": len(records)-failed, "failed_count": failed}, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
