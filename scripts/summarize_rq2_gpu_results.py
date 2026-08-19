#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from configfuzz.experiment import ExecutionMilestone, MILESTONE_ORDER


FRAMEWORK_ORDER = (
    "pytorch-cuda",
    "deepspeed",
    "transformers-accelerate",
    "megatron-core",
)
METHOD_ORDER = (
    "raw_mutation",
    "native_validator_guided",
    "constraint_filter_only",
    "static_hard_configfuzz",
    "configfuzz",
    "global_repair",
)
TARGET_MILESTONE = ExecutionMilestone.OPTIMIZER_STEP


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _deep(record: Mapping[str, Any]) -> bool:
    try:
        milestone = ExecutionMilestone(str(record.get("deepest_milestone", "unknown")))
    except ValueError:
        return False
    return MILESTONE_ORDER.get(milestone, -1) >= MILESTONE_ORDER[TARGET_MILESTONE]


def _entropy_bits(values: list[str]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _observed_behavior_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = set(str(value) for value in row.get("behavior_ids", ()))
    if not values:
        values = {
            *(f"branch:{value}" for value in row.get("runtime_branches", ())),
            *(f"topology:{value}" for value in row.get("topologies", ())),
            *(f"feature:{value}" for value in row.get("feature_interactions", ())),
            *(f"backend:{value}" for value in row.get("backend_paths", ())),
        }
    observed: list[str] = []
    for value in values:
        if value.startswith("branch:") and not value.startswith("branch:forward_path="):
            continue
        if value in {"feature:moe_configuration", "feature:shared_experts"}:
            continue
        if value.startswith(("branch:", "topology:", "feature:", "backend:")):
            observed.append(value)
    return tuple(sorted(set(observed)))


def _observed_signature(row: Mapping[str, Any]) -> str | None:
    values = _observed_behavior_ids(row)
    if not values:
        return None
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _diversity_summary(records: list[Mapping[str, Any]], accelerator_hours: float) -> dict[str, Any]:
    instrumented = [
        row
        for row in records
        if bool(row.get("generated")) and bool(_observed_behavior_ids(row))
    ]
    behavior_ids = {
        value
        for row in instrumented
        for value in _observed_behavior_ids(row)
    }
    signatures = [
        signature
        for row in instrumented
        if (signature := _observed_signature(row)) is not None
    ]

    runtime_branches = {
        value.removeprefix("branch:")
        for row in instrumented
        for value in _observed_behavior_ids(row)
        if value.startswith("branch:")
    }
    topologies = {
        value.removeprefix("topology:")
        for row in instrumented
        for value in _observed_behavior_ids(row)
        if value.startswith("topology:")
    }
    feature_interactions = {
        value.removeprefix("feature:")
        for row in instrumented
        for value in _observed_behavior_ids(row)
        if value.startswith("feature:")
    }
    backend_paths = {
        value.removeprefix("backend:")
        for row in instrumented
        for value in _observed_behavior_ids(row)
        if value.startswith("backend:")
    }

    return {
        "instrumented_execution_count": len(instrumented),
        "behavior_policy": "executed-path-v1",
        "runtime_branches": len(runtime_branches),
        "topologies": len(topologies),
        "feature_interactions": len(feature_interactions),
        "backend_paths": len(backend_paths),
        "runtime_behavior_ids": len(behavior_ids),
        "runtime_behavior_ids_per_accelerator_hour": (
            len(behavior_ids) / accelerator_hours if accelerator_hours > 0 else None
        ),
        "behavior_signatures": len(set(signatures)),
        "behavior_signatures_per_accelerator_hour": (
            len(set(signatures)) / accelerator_hours if accelerator_hours > 0 else None
        ),
        "behavior_signature_entropy_bits": _entropy_bits(signatures),
    }


def _method_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    generated = [row for row in records if bool(row.get("generated"))]
    deep = [row for row in records if _deep(row)]
    retained = [
        row
        for row in generated
        if bool(row.get("target_value_preserved"))
    ]
    gpu_seconds = sum(
        float(row.get("accelerator_seconds", row.get("gpu_seconds", 0.0)))
        for row in records
    )
    accelerator_hours = gpu_seconds / 3600.0
    return {
        "run_count": len(records),
        "generated_count": len(generated),
        "generation_rate": len(generated) / len(records) if records else 0.0,
        "ipde_count": len(deep),
        "ipde_rate": len(deep) / len(records) if records else 0.0,
        "deep_execution_count": len(deep),
        "deep_execution_rate": len(deep) / len(records) if records else 0.0,
        "target_retention_rate": len(retained) / len(generated) if generated else 1.0,
        "coordinated_case_count": sum(bool(row.get("coordinated_parameters")) for row in records),
        "accelerator_seconds": gpu_seconds,
        "ipde_per_accelerator_hour": (
            len(deep) / accelerator_hours if accelerator_hours > 0 else None
        ),
        "diversity": _diversity_summary(records, accelerator_hours),
        "outcome_counts": dict(sorted(Counter(str(row.get("outcome", "unknown")) for row in records).items())),
    }


def _paired_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_intent: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        intent_id = row.get("intent_id")
        method = row.get("method")
        if intent_id and method in {"raw_mutation", "configfuzz"}:
            by_intent[str(intent_id)][str(method)] = row
    counts = Counter()
    raw_targets = Counter()
    cf_targets = Counter()
    cf_only_cases: list[dict[str, Any]] = []
    raw_only_cases: list[dict[str, Any]] = []
    for intent_id, methods in sorted(by_intent.items()):
        raw = methods.get("raw_mutation")
        cf = methods.get("configfuzz")
        if raw is None or cf is None:
            continue
        raw_deep = _deep(raw)
        cf_deep = _deep(cf)
        if raw_deep and cf_deep:
            counts["both_deep"] += 1
        elif raw_deep:
            counts["raw_only"] += 1
            target = str(cf.get("metadata", {}).get("target_parameter", "unknown"))
            raw_targets[target] += 1
            raw_only_cases.append(_paired_case(intent_id, raw, cf))
        elif cf_deep:
            counts["configfuzz_only"] += 1
            target = str(cf.get("metadata", {}).get("target_parameter", "unknown"))
            cf_targets[target] += 1
            cf_only_cases.append(_paired_case(intent_id, raw, cf))
        else:
            counts["neither"] += 1
    return {
        "both_deep": counts["both_deep"],
        "raw_only": counts["raw_only"],
        "configfuzz_only": counts["configfuzz_only"],
        "neither": counts["neither"],
        "raw_only_target_counts": dict(sorted(raw_targets.items())),
        "configfuzz_only_target_counts": dict(sorted(cf_targets.items())),
        "configfuzz_only_cases": cf_only_cases,
        "raw_only_cases": raw_only_cases,
    }


def _paired_case(
    intent_id: str,
    raw: Mapping[str, Any],
    cf: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "intent_id": intent_id,
        "workload_id": cf.get("workload_id"),
        "target_parameter": cf.get("metadata", {}).get("target_parameter"),
        "target_value": cf.get("metadata", {}).get("target_value"),
        "coordinated_parameters": list(cf.get("coordinated_parameters", ())),
        "solver_modifications": dict(cf.get("solver_modifications", {})),
        "raw_deepest_milestone": raw.get("deepest_milestone"),
        "raw_outcome": raw.get("outcome"),
        "configfuzz_deepest_milestone": cf.get("deepest_milestone"),
        "configfuzz_outcome": cf.get("outcome"),
    }


def summarize(directory: Path) -> dict[str, Any]:
    all_rows: dict[str, list[dict[str, Any]]] = {}
    framework_payload: dict[str, Any] = {}
    aggregate_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    paired: dict[str, Any] = {}
    total_records = 0
    reuse_rows = 0
    reused_source_runs: set[str] = set()
    for framework in FRAMEWORK_ORDER:
        path = directory / f"{framework}.jsonl"
        rows = _read_jsonl(path)
        all_rows[framework] = rows
        total_records += len(rows)
        by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            method = str(row.get("method"))
            by_method[method].append(row)
            aggregate_by_method[method].append(row)
            metadata = row.get("metadata")
            if isinstance(metadata, Mapping) and isinstance(metadata.get("runtime_reuse"), Mapping):
                reuse_rows += 1
                source_run = metadata["runtime_reuse"].get("source_run_id")
                if source_run:
                    reused_source_runs.add(str(source_run))
        framework_payload[framework] = {
            "intent_count": len({str(row.get("intent_id")) for row in rows if row.get("intent_id")}),
            "record_count": len(rows),
            "methods": {
                method: _method_summary(by_method.get(method, ()))
                for method in METHOD_ORDER
            },
        }
        paired[framework] = _paired_summary(rows)

    aggregate = {
        method: _method_summary(aggregate_by_method.get(method, ()))
        for method in METHOD_ORDER
    }
    paired_total = Counter()
    for framework in FRAMEWORK_ORDER:
        for key in ("both_deep", "raw_only", "configfuzz_only", "neither"):
            paired_total[key] += int(paired[framework][key])
    paired["aggregate"] = dict(paired_total)
    return {
        "schema_version": 3,
        "name": "rq2-gpu-final-primary-seed",
        "seed": 2026,
        "target_milestone": TARGET_MILESTONE.value,
        "total_records": total_records,
        "frameworks": framework_payload,
        "aggregate": aggregate,
        "paired_raw_vs_configfuzz": paired,
        "runtime_reuse": {
            "reused_result_rows": reuse_rows,
            "unique_source_runs": len(reused_source_runs),
            "note": (
                "Runtime reuse is admitted only for byte-equivalent canonical materialized "
                "configurations under the same framework, seed, launcher, and unchanged execution harness."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.directory)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_records": payload["total_records"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
