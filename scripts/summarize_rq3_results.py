#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

FRAMEWORKS = ("deepspeed", "transformers-accelerate", "megatron-core", "pytorch-cuda")
METHODS = (
    "raw_mutation",
    "native_validator_guided",
    "constraint_filter_only",
    "static_hard_configfuzz",
    "configfuzz",
    "global_repair",
)
_EXCEPTION_RE = re.compile(
    r"((?:ValueError|RuntimeError|AssertionError|ZeroDivisionError|IndexError|KeyError|"
    r"TypeError|NotImplementedError|AttributeError|FloatingPointError):.*)"
)
_NUMBER_RE = re.compile(r"\b-?\d+(?:\.\d+)?\b")
_PATH_RE = re.compile(r"/[^\s:\"]+(?:/[^\s:\"]+)+")
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a completed ConfigFuzz RQ3 current-version campaign.")
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            rows.append(payload)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_key(framework: str, row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    metadata = row.get("metadata") or {}
    return (
        framework,
        str(row.get("workload_id")),
        str(metadata.get("target_parameter")),
        json.dumps(metadata.get("target_value"), ensure_ascii=False, sort_keys=True),
    )


def normalized_failure_signature(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    log_value = metadata.get("log_path")
    text = ""
    if log_value:
        log_path = Path(str(log_value))
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    for line in text.splitlines():
        match = _EXCEPTION_RE.search(line)
        if match and "ChildFailedError" not in match.group(1):
            errors.append(match.group(1).strip())
    signature = errors[-1] if errors else f"no-explicit-exception::{row.get('deepest_milestone')}"
    signature = _PATH_RE.sub("<PATH>", signature)
    signature = _HEX_RE.sub("<HEX>", signature)
    signature = _NUMBER_RE.sub("<N>", signature)
    return re.sub(r"\s+", " ", signature).strip()[:1000]


def write_csv(path: Path, fieldnames: list[str], records: list[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    root = args.current_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    by_framework: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[tuple[str, dict[str, Any]]] = []
    source_files: dict[str, dict[str, Any]] = {}
    for framework in FRAMEWORKS:
        path = root / f"{framework}.jsonl"
        rows = load_rows(path)
        by_framework[framework] = rows
        all_rows.extend((framework, row) for row in rows)
        source_files[path.name] = {"records": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}

    framework_records: list[dict[str, Any]] = []
    for framework in FRAMEWORKS:
        rows = by_framework[framework]
        outcomes = Counter(str(row.get("outcome")) for row in rows)
        groups = {
            candidate_key(framework, row)
            for row in rows
            if row.get("outcome") == "unexplained_failure"
        }
        framework_records.append(
            {
                "framework": framework,
                "tests": len(rows),
                "valid": outcomes["valid"],
                "expected_rejection": outcomes["expected_rejection"],
                "unexplained_failure": outcomes["unexplained_failure"],
                "unknown": outcomes["unknown"],
                "resource_failure": outcomes["resource_failure"],
                "infrastructure_failure": outcomes["infrastructure_failure"],
                "deduplicated_unexplained_candidates": len(groups),
            }
        )
    write_csv(
        out / "current-version-by-framework.csv",
        [
            "framework", "tests", "valid", "expected_rejection", "unexplained_failure", "unknown",
            "resource_failure", "infrastructure_failure", "deduplicated_unexplained_candidates",
        ],
        framework_records,
    )

    method_records: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [(fw, row) for fw, row in all_rows if row.get("method") == method]
        outcomes = Counter(str(row.get("outcome")) for _, row in selected)
        deep = sum(
            str(row.get("deepest_milestone"))
            in {"forward", "backward", "optimizer_step", "checkpoint_save_load", "completed"}
            for _, row in selected
        )
        groups = {
            candidate_key(fw, row)
            for fw, row in selected
            if row.get("outcome") == "unexplained_failure"
        }
        method_records.append(
            {
                "method": method,
                "tests": len(selected),
                "valid": outcomes["valid"],
                "expected_rejection": outcomes["expected_rejection"],
                "unexplained_failure": outcomes["unexplained_failure"],
                "unknown": outcomes["unknown"],
                "resource_failure": outcomes["resource_failure"],
                "infrastructure_failure": outcomes["infrastructure_failure"],
                "deep_execution": deep,
                "deduplicated_unexplained_candidates": len(groups),
            }
        )
    write_csv(
        out / "current-version-by-method.csv",
        [
            "method", "tests", "valid", "expected_rejection", "unexplained_failure", "unknown",
            "resource_failure", "infrastructure_failure", "deep_execution", "deduplicated_unexplained_candidates",
        ],
        method_records,
    )

    priority = {method: index for index, method in enumerate(("configfuzz", "global_repair", "static_hard_configfuzz", "constraint_filter_only", "native_validator_guided", "raw_mutation"))}
    representatives: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    multiplicities: Counter[tuple[str, str, str, str]] = Counter()
    methods_per_candidate: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for framework, row in all_rows:
        if row.get("outcome") != "unexplained_failure":
            continue
        key = candidate_key(framework, row)
        multiplicities[key] += 1
        methods_per_candidate[key].add(str(row.get("method")))
        current = representatives.get(key)
        if current is None or priority.get(str(row.get("method")), 99) < priority.get(str(current.get("method")), 99):
            representatives[key] = row

    clusters: dict[tuple[str, str], list[tuple[tuple[str, str, str, str], dict[str, Any]]]] = defaultdict(list)
    for key, row in representatives.items():
        clusters[(key[0], normalized_failure_signature(row))].append((key, row))

    cluster_records: list[dict[str, Any]] = []
    for (framework, signature), entries in sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
        key, row = entries[0]
        metadata = row.get("metadata") or {}
        cluster_records.append(
            {
                "framework": framework,
                "candidate_count": len(entries),
                "normalized_signature": signature,
                "representative_run_id": row.get("run_id"),
                "representative_workload": row.get("workload_id"),
                "representative_target_parameter": metadata.get("target_parameter"),
                "representative_target_value": json.dumps(metadata.get("target_value"), ensure_ascii=False, sort_keys=True),
            }
        )
    write_csv(
        out / "unexplained-signature-clusters.csv",
        [
            "framework", "candidate_count", "normalized_signature", "representative_run_id",
            "representative_workload", "representative_target_parameter", "representative_target_value",
        ],
        cluster_records,
    )

    candidate_records: list[dict[str, Any]] = []
    for key, row in sorted(representatives.items()):
        metadata = row.get("metadata") or {}
        candidate_records.append(
            {
                "framework": key[0],
                "workload": key[1],
                "target_parameter": key[2],
                "target_value": key[3],
                "row_multiplicity": multiplicities[key],
                "methods": ";".join(sorted(methods_per_candidate[key])),
                "representative_run_id": row.get("run_id"),
                "deepest_milestone": row.get("deepest_milestone"),
                "normalized_signature": normalized_failure_signature(row),
            }
        )
    write_csv(
        out / "unexplained-candidates.csv",
        [
            "framework", "workload", "target_parameter", "target_value", "row_multiplicity", "methods",
            "representative_run_id", "deepest_milestone", "normalized_signature",
        ],
        candidate_records,
    )

    outcomes = Counter(str(row.get("outcome")) for _, row in all_rows)
    summary = {
        "schema_version": 1,
        "campaign": "rq3-current-version-gpu45-20260820",
        "tests": len(all_rows),
        "frameworks": list(FRAMEWORKS),
        "methods": list(METHODS),
        "outcomes": dict(sorted(outcomes.items())),
        "deduplicated_unexplained_candidates": len(representatives),
        "normalized_failure_signature_clusters": len(clusters),
        "source_files": source_files,
        "notes": [
            "unexplained_failure is an executor triage bucket, not a bug count",
            "deduplicated candidates group rows by framework/workload/target parameter/target value across methods",
            "normalized failure clusters are diagnostic groupings and are not independent root-cause counts",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
