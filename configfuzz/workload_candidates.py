from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


def materialize_workload_candidates(
    source_spec_path: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
    registry_output: str | Path,
) -> dict[str, Any]:
    spec_path = Path(source_spec_path).expanduser().resolve()
    root = Path(source_root).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    registry_path = Path(registry_output).expanduser().resolve()
    if not (root / ".git").is_dir():
        raise ValueError(f"source root is not a Git repository: {root}")
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("workload-candidate source spec must be an object")
    source_repository = raw.get("source_repository")
    if not isinstance(source_repository, Mapping):
        raise ValueError("source_repository must be an object")
    expected_revision = _required_string(source_repository, "revision")
    actual_revision = _git_output(root, "rev-parse", "HEAD")
    if actual_revision is None:
        raise ValueError(f"cannot resolve source revision: {root}")
    if bool(source_repository.get("required_clean_revision", False)):
        if actual_revision != expected_revision:
            raise ValueError(
                f"source revision mismatch: expected {expected_revision}, got {actual_revision}"
            )
        if _git_output(root, "status", "--porcelain"):
            raise ValueError(
                "source repository must be clean for candidate materialization"
            )

    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate source spec must contain candidates")
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    workload_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("candidate entry must be an object")
        candidate_id = _required_string(raw_candidate, "candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate workload candidate: {candidate_id}")
        seen.add(candidate_id)
        model_relative = Path(_required_string(raw_candidate, "model_config"))
        command_relative = Path(_required_string(raw_candidate, "command_template"))
        model_path = (root / model_relative).resolve()
        command_path = (root / command_relative).resolve()
        _require_beneath(root, model_path)
        _require_beneath(root, command_path)
        if not model_path.is_file() or not command_path.is_file():
            raise FileNotFoundError(
                f"candidate {candidate_id}: missing model config or command template"
            )
        model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        if not isinstance(model_config, Mapping):
            raise ValueError(
                f"candidate {candidate_id}: model config must be an object"
            )
        baseline = dict(model_config)
        world_size = int(raw_candidate.get("world_size", 1))
        command_text = command_path.read_text(encoding="utf-8")
        command_overrides = _extract_command_overrides(command_text)
        effective_config, effective_sources = _build_effective_config(
            model_config,
            command_overrides,
            world_size=world_size,
        )
        baseline["world_size"] = world_size
        baseline["command_overrides"] = command_overrides
        baseline["effective_config"] = effective_config
        baseline["effective_config_sources"] = effective_sources
        snapshot_path = out_dir / f"{candidate_id}.json"
        snapshot_path.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        snapshot_relative = Path(
            Path(snapshot_path).relative_to(registry_path.parent)
            if snapshot_path.is_relative_to(registry_path.parent)
            else snapshot_path
        )
        model_hash = _sha256(model_path)
        command_hash = _sha256(command_path)
        snapshot_hash = _sha256(snapshot_path)
        manifest_records.append(
            {
                **dict(raw_candidate),
                "source_revision": actual_revision,
                "model_config_sha256": model_hash,
                "command_template_sha256": command_hash,
                "baseline_snapshot": str(snapshot_relative).replace("\\", "/"),
                "baseline_snapshot_sha256": snapshot_hash,
                "command_override_count": len(command_overrides),
                "effective_parameter_count": len(effective_config),
                "status": "source_selected_unverified",
            }
        )
        workload_records.append(
            {
                "workload_id": candidate_id,
                "family": _required_string(raw_candidate, "family"),
                "baseline_id": _required_string(raw_candidate, "baseline_id"),
                "baseline_config": str(snapshot_relative).replace("\\", "/"),
                "metadata": {
                    "target_workload_id": _required_string(
                        raw_candidate, "target_workload_id"
                    ),
                    "priority": str(raw_candidate.get("priority", "primary")),
                    "minimum_intents": int(raw_candidate.get("minimum_intents", 0)),
                    "status": "source_selected_unverified",
                    "source_model_config": str(model_relative).replace("\\", "/"),
                    "source_command_template": str(command_relative).replace("\\", "/"),
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "name": "rq2-workload-candidates",
        "source_repository": {
            "label": str(source_repository.get("label", "source")),
            "revision": actual_revision,
            "clean": not bool(_git_output(root, "status", "--porcelain")),
        },
        "candidate_count": len(manifest_records),
        "primary_candidate_count": sum(
            str(item.get("priority", "primary")) == "primary"
            for item in manifest_records
        ),
        "policy": {
            "candidate_is_not_bound_workload": True,
            "accelerator_validation_required": True,
            "final_binding_requires_optimizer_step_and_checkpoint": True,
        },
        "candidates": manifest_records,
    }
    manifest_path = registry_path.with_name("workload_candidates.yaml")
    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    registry = {
        "schema_version": 1,
        "name": "rq2-workload-candidate-registry",
        "metadata": {
            "status": "source_selected_unverified",
            "source_revision": actual_revision,
            "do_not_use_for_final_campaign_until_bound": True,
        },
        "workloads": workload_records,
    }
    registry_path.write_text(
        yaml.safe_dump(registry, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "registry": registry,
        "manifest_path": str(manifest_path),
        "registry_path": str(registry_path),
    }


def validate_workload_candidate_manifest(
    manifest_path: str | Path,
    registry_path: str | Path,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    registry_file = Path(registry_path).expanduser().resolve()
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    registry = yaml.safe_load(registry_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(registry, Mapping):
        raise ValueError("candidate manifest and registry must be objects")
    candidates = manifest.get("candidates")
    workloads = registry.get("workloads")
    if not isinstance(candidates, list) or not isinstance(workloads, list):
        raise ValueError("candidate manifest and registry records must be lists")
    by_id = {
        str(item.get("candidate_id")): item
        for item in candidates
        if isinstance(item, Mapping)
    }
    workload_by_id = {
        str(item.get("workload_id")): item
        for item in workloads
        if isinstance(item, Mapping)
    }
    if set(by_id) != set(workload_by_id):
        raise ValueError("candidate manifest and workload registry IDs differ")
    primary_roles: set[str] = set()
    for candidate_id, candidate in by_id.items():
        if candidate.get("status") != "source_selected_unverified":
            raise ValueError(f"{candidate_id}: invalid candidate status")
        snapshot_value = candidate.get("baseline_snapshot")
        snapshot = Path(str(snapshot_value))
        if not snapshot.is_absolute():
            snapshot = (manifest_file.parent / snapshot).resolve()
        if not snapshot.is_file():
            raise FileNotFoundError(f"{candidate_id}: baseline snapshot missing")
        if _sha256(snapshot) != str(candidate.get("baseline_snapshot_sha256", "")):
            raise ValueError(f"{candidate_id}: baseline snapshot hash mismatch")
        blockers = candidate.get("blockers")
        if not isinstance(blockers, list) or not blockers:
            raise ValueError(f"{candidate_id}: unresolved blockers must be recorded")
        if str(candidate.get("priority", "primary")) == "primary":
            primary_roles.add(str(candidate.get("target_workload_id", "")))
    required_roles = {
        "qwen2-train",
        "llama2-train",
        "chatglm3-train",
        "mixtral-train",
        "deepseekv3-train",
        "internvl3-train",
        "cogvideox-train",
    }
    if primary_roles != required_roles:
        raise ValueError(
            f"primary candidates do not cover required roles: {sorted(primary_roles)}"
        )
    return {
        "valid": True,
        "candidate_count": len(by_id),
        "primary_candidate_count": sum(
            str(item.get("priority", "primary")) == "primary" for item in by_id.values()
        ),
        "primary_roles": sorted(primary_roles),
    }


def _git_output(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{key} must be a non-empty string")
    return str(value).strip()


def _require_beneath(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source path escapes repository root: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_DEFAULT_EFFECTIVE_VALUES: dict[str, Any] = {
    "context_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "sequence_parallel": False,
    "distribute_saved_activations": False,
    "num_layers_per_virtual_pipeline_stage": None,
    "recompute_granularity": None,
    "recompute_method": None,
    "recompute_num_layers": None,
}


def _extract_command_overrides(command_text: str) -> dict[str, Any]:
    variables = _extract_shell_assignments(command_text)
    tokens: list[str] = []
    for name, value in variables.items():
        if not name.endswith("_ARGS"):
            continue
        expanded = _expand_shell_variables(value, variables)
        expanded = re.sub(r"\\[ \t]*\r?\n", " ", expanded)
        try:
            tokens.extend(shlex.split(expanded, comments=True, posix=True))
        except ValueError:
            continue

    overrides: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--") or token == "--":
            index += 1
            continue
        raw_name = token[2:]
        if "=" in raw_name:
            raw_name, raw_value = raw_name.split("=", 1)
            value: Any = _parse_shell_scalar(raw_value)
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            index += 1
            value = _parse_shell_scalar(tokens[index])
        else:
            value = True

        if raw_name.startswith("no-"):
            name = raw_name[3:].replace("-", "_")
            value = False
        else:
            name = raw_name.replace("-", "_")
        overrides[name] = value
        index += 1

    _add_command_aliases(overrides)
    return dict(sorted(overrides.items()))


def _extract_shell_assignments(command_text: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    quoted = re.compile(r"(?ms)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\"'])(.*?)\2\s*$")
    occupied: list[tuple[int, int]] = []
    for match in quoted.finditer(command_text):
        assignments[match.group(1)] = match.group(3)
        occupied.append(match.span())

    def inside_quoted_assignment(position: int) -> bool:
        return any(start <= position < end for start, end in occupied)

    simple = re.compile(
        r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s#]+|\"[^\"]*\"|'[^']*')\s*(?:#.*)?$"
    )
    for match in simple.finditer(command_text):
        if inside_quoted_assignment(match.start()):
            continue
        value = match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        assignments[match.group(1)] = value
    return assignments


def _expand_shell_variables(value: str, variables: Mapping[str, str]) -> str:
    result = value
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
    for _ in range(8):
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            name = match.group(1) or match.group(2)
            replacement = variables.get(name)
            if replacement is None:
                return match.group(0)
            changed = True
            return replacement

        result = pattern.sub(replace, result)
        if not changed:
            break
    return result


def _parse_shell_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            pass
    return text


def _add_command_aliases(overrides: dict[str, Any]) -> None:
    if "disable_bias_linear" in overrides:
        overrides["add_bias_linear"] = not bool(overrides["disable_bias_linear"])
    if "swiglu" in overrides:
        overrides["gated_linear_unit"] = bool(overrides["swiglu"])
    if "use_flash_attn" in overrides:
        overrides["use_flash_attention"] = overrides["use_flash_attn"]
    if "seq_length" in overrides:
        overrides["seq_len"] = overrides["seq_length"]
    if "num_experts" in overrides:
        overrides["num_moe_experts"] = overrides["num_experts"]
    if "topk_group" in overrides:
        overrides["moe_router_group_topk"] = overrides["topk_group"]


def _build_effective_config(
    model_config: Mapping[str, Any],
    command_overrides: Mapping[str, Any],
    *,
    world_size: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    effective: dict[str, Any] = {}
    sources: dict[str, str] = {}

    def record(name: str, value: Any, source: str, *, overwrite: bool = True) -> None:
        if not overwrite and name in effective:
            return
        parsed = _parse_shell_scalar(value) if isinstance(value, str) else value
        effective[name] = parsed
        sources[name] = source

    for name, value in _DEFAULT_EFFECTIVE_VALUES.items():
        record(name, value, "framework_default")

    section_order = (
        "TransformerConfig",
        "MLATransformerConfig",
        "extra_config",
        "get_gpt_layer_local_spec",
    )
    for section in section_order:
        section_value = model_config.get(section)
        if not isinstance(section_value, Mapping):
            continue
        for name, value in _scalar_leaves(section_value):
            record(name, value, f"model_config:{section}")

    for name, value in command_overrides.items():
        record(name, value, "command_template")

    record("world_size", world_size, "candidate_spec")
    _derive_effective_values(effective, sources)
    return dict(sorted(effective.items())), dict(sorted(sources.items()))


def _scalar_leaves(value: Mapping[str, Any]) -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    for name, item in value.items():
        if isinstance(item, Mapping):
            leaves.extend(_scalar_leaves(item))
        elif not isinstance(item, (list, tuple, set, dict)):
            leaves.append((str(name), item))
    return leaves


def _derive_effective_values(
    effective: dict[str, Any], sources: dict[str, str]
) -> None:
    def alias(target: str, source: str) -> None:
        if target not in effective and source in effective:
            effective[target] = effective[source]
            sources[target] = f"derived_alias:{source}"

    alias("use_flash_attention", "use_flash_attn")
    alias("seq_len", "seq_length")
    alias("num_experts", "num_moe_experts")
    alias("num_moe_experts", "num_experts")
    alias("swiglu", "gated_linear_unit")

    if "group_query_attention" not in effective:
        heads = effective.get("num_attention_heads")
        groups = effective.get("num_query_groups")
        if isinstance(heads, int) and isinstance(groups, int):
            effective["group_query_attention"] = groups < heads
            sources["group_query_attention"] = (
                "derived:num_query_groups<num_attention_heads"
            )

    factors = [
        effective.get("tensor_model_parallel_size", 1),
        effective.get("pipeline_model_parallel_size", 1),
        effective.get("context_parallel_size", 1),
    ]
    if all(isinstance(item, int) and item > 0 for item in factors):
        model_parallel = int(factors[0]) * int(factors[1]) * int(factors[2])
        world_size = effective.get("world_size")
        if (
            isinstance(world_size, int)
            and world_size > 0
            and world_size % model_parallel == 0
        ):
            effective["data_parallel_size"] = world_size // model_parallel
            sources["data_parallel_size"] = "derived:world_size/(tp*pp*cp)"
