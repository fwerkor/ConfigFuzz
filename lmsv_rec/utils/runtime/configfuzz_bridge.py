from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


_SECTION_HINTS = {
    "tensor_model_parallel_size": "parallel",
    "pipeline_model_parallel_size": "parallel",
    "expert_model_parallel_size": "parallel",
    "context_parallel_size": "parallel",
    "sequence_parallel": "parallel",
    "num_layers_per_virtual_pipeline_stage": "parallel",
    "micro_batch_size": "training",
    "global_batch_size": "training",
    "train_iters": "training",
    "lr": "training",
    "min_lr": "training",
    "weight_decay": "training",
    "clip_grad": "training",
    "lr_warmup_fraction": "training",
    "lr_warmup_iters": "training",
    "lr_decay_style": "training",
    "num_experts": "moe",
    "moe_router_topk": "moe",
    "moe_router_load_balancing_type": "moe",
    "moe_router_pre_softmax": "moe",
    "moe_intermediate_size": "moe",
    "q_lora_rank": "mla",
    "kv_lora_rank": "mla",
    "qk_rope_head_dim": "mla",
    "qk_nope_head_dim": "mla",
    "v_head_dim": "mla",
}

_ALIASES = {
    "tensor_parallel_size": "tensor_model_parallel_size",
    "pipeline_parallel_size": "pipeline_model_parallel_size",
    "expert_parallel_size": "expert_model_parallel_size",
    "num_moe_experts": "num_experts",
    "expert_num": "num_experts",
    "per_token_num_experts_chosen": "moe_router_topk",
    "num_experts_chosen": "moe_router_topk",
}


def load_configfuzz_assignments(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ConfigFuzz assignments must be a JSON object")
    for key in ("assignments", "config", "configuration"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            payload = nested
            break
    return {str(key): copy.deepcopy(value) for key, value in payload.items()}


def apply_configfuzz_assignments(
    configuration: Mapping[str, Any],
    assignments: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = copy.deepcopy(dict(configuration))
    applied: dict[str, Any] = {}

    for raw_name, value in assignments.items():
        name = _ALIASES.get(str(raw_name), str(raw_name))
        if isinstance(value, Mapping) and name in updated:
            section = updated[name]
            if not isinstance(section, dict):
                raise ValueError(f"ConfigFuzz section {name!r} is not a mapping")
            for child_name, child_value in value.items():
                canonical = _ALIASES.get(str(child_name), str(child_name))
                section[canonical] = copy.deepcopy(child_value)
                applied[f"{name}.{canonical}"] = copy.deepcopy(child_value)
            continue

        path = _resolve_assignment_path(updated, name)
        _set_path(updated, path, copy.deepcopy(value))
        applied[path] = copy.deepcopy(value)

    return updated, applied


def apply_configfuzz_assignments_file(
    configuration: Mapping[str, Any],
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return apply_configfuzz_assignments(
        configuration,
        load_configfuzz_assignments(path),
    )


def find_configfuzz_assignment_mismatches(
    configuration: Mapping[str, Any],
    applied: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    mismatches: dict[str, dict[str, Any]] = {}
    for path, expected in applied.items():
        actual = _get_path(configuration, path)
        if actual != expected:
            mismatches[path] = {"expected": expected, "actual": actual}
    return mismatches


def ensure_configfuzz_assignments_preserved(
    configuration: Mapping[str, Any],
    applied: Mapping[str, Any],
) -> None:
    mismatches = find_configfuzz_assignment_mismatches(configuration, applied)
    if not mismatches:
        return
    details = ", ".join(
        f"{path}: {values['expected']!r} -> {values['actual']!r}"
        for path, values in sorted(mismatches.items())
    )
    raise ValueError(f"ConfigFuzz assignments were repaired: {details}")


def _resolve_assignment_path(configuration: Mapping[str, Any], name: str) -> str:
    if "." in name:
        if _path_exists(configuration, name):
            return name
        section, _, leaf = name.partition(".")
        if section in configuration and isinstance(configuration[section], Mapping):
            return name
        raise KeyError(f"unknown ConfigFuzz assignment path: {name}")

    matches = _leaf_paths(configuration, name)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preferred = _SECTION_HINTS.get(name)
        preferred_path = f"{preferred}.{name}" if preferred else ""
        if preferred_path in matches:
            return preferred_path
        raise ValueError(
            f"ambiguous ConfigFuzz assignment {name!r}: {', '.join(matches)}"
        )

    section = _SECTION_HINTS.get(name, "model")
    if section not in configuration:
        if not isinstance(configuration, dict):
            raise ValueError("configuration must be mutable")
        configuration[section] = {}
    if not isinstance(configuration[section], Mapping):
        raise ValueError(f"configuration section {section!r} is not a mapping")
    return f"{section}.{name}"


def _leaf_paths(value: Mapping[str, Any], leaf: str, prefix: str = "") -> list[str]:
    matches: list[str] = []
    for raw_key, item in value.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if key == leaf:
            matches.append(path)
        if isinstance(item, Mapping):
            matches.extend(_leaf_paths(item, leaf, path))
    return matches


def _path_exists(value: Mapping[str, Any], path: str) -> bool:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _set_path(value: dict[str, Any], path: str, new_value: Any) -> None:
    parts = path.split(".")
    current = value
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"ConfigFuzz assignment parent {part!r} is not a mapping")
        current = child
    current[parts[-1]] = new_value
