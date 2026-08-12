from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from configfuzz.experiment import MutationIntent


_GRID_CLASSES = {
    "integer_adjacent_below",
    "integer_adjacent_above",
    "integer_scale_boundary",
    "integer_common_boundary",
    "integer_power_of_two_boundary",
    "integer_divisibility_boundary",
    "float_zero_neighborhood",
    "float_scale_boundary",
    "float_adjacent_below",
    "float_adjacent_above",
    "float_sign_boundary",
    "float_zero_boundary",
    "boolean_transition",
}


def select_balanced_intents(
    candidate_path: str | Path,
    workload_registry_path: str | Path,
    *,
    default_per_primary_workload: int = 300,
    intent_pool: str = "method_independent",
) -> dict[str, Any]:
    candidate_file = Path(candidate_path).expanduser().resolve()
    registry_file = Path(workload_registry_path).expanduser().resolve()
    candidate_root = yaml.safe_load(candidate_file.read_text(encoding="utf-8"))
    registry_root = yaml.safe_load(registry_file.read_text(encoding="utf-8"))
    if not isinstance(candidate_root, Mapping) or not isinstance(
        registry_root, Mapping
    ):
        raise ValueError("candidate intent file and workload registry must be objects")

    raw_intents = candidate_root.get("intents")
    raw_workloads = registry_root.get("workloads")
    if not isinstance(raw_intents, list) or not isinstance(raw_workloads, list):
        raise ValueError("candidate intents and workload records must be lists")
    intents = [
        MutationIntent.from_dict(_mapping(item, "mutation intent"))
        for item in raw_intents
    ]
    intents = [item for item in intents if item.intent_pool == intent_pool]

    workload_records: dict[str, Mapping[str, Any]] = {}
    for item in raw_workloads:
        record = _mapping(item, "workload")
        workload_id = str(record.get("workload_id", "")).strip()
        if not workload_id or workload_id in workload_records:
            raise ValueError(f"invalid or duplicate workload id: {workload_id!r}")
        workload_records[workload_id] = record

    by_workload: dict[str, list[MutationIntent]] = defaultdict(list)
    for intent in intents:
        by_workload[intent.workload_id].append(intent)

    selected: list[MutationIntent] = []
    selection_stats: dict[str, Any] = {}
    for workload_id, record in sorted(workload_records.items()):
        metadata = _mapping(record.get("metadata", {}), "workload metadata")
        if str(metadata.get("priority", "primary")) != "primary":
            continue
        requested = int(metadata.get("minimum_intents", default_per_primary_workload))
        if requested <= 0:
            requested = default_per_primary_workload
        candidates = by_workload.get(workload_id, [])
        if len(candidates) < requested:
            raise ValueError(
                f"workload {workload_id}: requested {requested} intents but only "
                f"{len(candidates)} candidates are available"
            )
        workload_selected = _round_robin_parameter_selection(candidates, requested)
        selected.extend(workload_selected)
        selection_stats[workload_id] = {
            "candidate_count": len(candidates),
            "selected_count": len(workload_selected),
            "parameter_count": len(
                {item.target_parameter for item in workload_selected}
            ),
            "intent_class_count": len(
                {item.intent_class for item in workload_selected}
            ),
            "constraint_count": len(
                {
                    constraint_id
                    for item in workload_selected
                    for constraint_id in item.source_constraint_ids
                }
            ),
        }

    selected.sort(key=lambda item: item.intent_id)
    _validate_unique_targets(selected)
    source_sha256 = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "name": "rq2-balanced-candidate-intents",
        "metadata": {
            "status": "accelerator_unverified_frozen_candidate",
            "source_candidate_file": candidate_file.name,
            "source_candidate_sha256": source_sha256,
            "intent_pool": intent_pool,
            "selection_policy": (
                "deterministic parameter round-robin within the selected intent pool"
            ),
            "workload_selection": selection_stats,
            "intent_count": len(selected),
            "warning": (
                "The workload baselines remain accelerator-unverified. Regenerate and freeze a "
                "final set after optimizer-step and checkpoint validation."
            ),
        },
        "intents": [item.to_dict() for item in selected],
    }


def _round_robin_parameter_selection(
    candidates: Sequence[MutationIntent], requested: int
) -> list[MutationIntent]:
    buckets: dict[str, deque[MutationIntent]] = {}
    grouped: dict[str, list[MutationIntent]] = defaultdict(list)
    for intent in candidates:
        grouped[intent.target_parameter].append(intent)
    for parameter, items in grouped.items():
        items.sort(key=_intent_priority_key)
        buckets[parameter] = deque(items)

    selected: list[MutationIntent] = []
    parameters = sorted(buckets)
    while len(selected) < requested:
        made_progress = False
        for parameter in parameters:
            bucket = buckets[parameter]
            if not bucket:
                continue
            selected.append(bucket.popleft())
            made_progress = True
            if len(selected) == requested:
                break
        if not made_progress:
            break
    if len(selected) != requested:
        raise ValueError(
            f"balanced selection exhausted candidates: requested {requested}, got {len(selected)}"
        )
    return selected


def _intent_priority_key(intent: MutationIntent) -> tuple[int, str, str]:
    grid_rank = 1 if intent.intent_class in _GRID_CLASSES else 0
    value_key = json.dumps(intent.target_value, ensure_ascii=False, sort_keys=True)
    return grid_rank, intent.intent_class, value_key


def _validate_unique_targets(intents: Sequence[MutationIntent]) -> None:
    seen_ids: set[str] = set()
    seen_targets: set[tuple[str, str, str]] = set()
    for intent in intents:
        if intent.intent_id in seen_ids:
            raise ValueError(f"duplicate intent id: {intent.intent_id}")
        seen_ids.add(intent.intent_id)
        value_key = json.dumps(intent.target_value, ensure_ascii=False, sort_keys=True)
        key = (intent.workload_id, intent.target_parameter, value_key)
        if key in seen_targets:
            raise ValueError(f"duplicate workload/parameter/value intent: {key}")
        seen_targets.add(key)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value
