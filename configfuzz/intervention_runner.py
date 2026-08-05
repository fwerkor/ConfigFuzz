from __future__ import annotations

import copy
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from configfuzz.outcomes import ClassificationPolicy, OutcomeLabel, classify_observation
from configfuzz.probing import ProbeSample, execute_command


@dataclass(frozen=True, slots=True)
class InterventionExecutionManifest:
    baseline_config: Path
    command: tuple[str, ...]
    cwd: Path = Path(".")
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    classification: ClassificationPolicy = field(default_factory=ClassificationPolicy)
    provenance_patterns: tuple[str, ...] = ()
    roles: tuple[str, ...] = ("satisfying", "violating", "repaired")

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("intervention command must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self.roles:
            raise ValueError("at least one intervention role is required")

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "InterventionExecutionManifest":
        raw_command = data.get("command")
        if not isinstance(raw_command, list) or not raw_command:
            raise ValueError("intervention manifest command must be a non-empty array")
        if data.get("baseline_config") is None:
            raise ValueError("intervention manifest requires baseline_config")
        baseline_config = Path(str(data["baseline_config"]))
        cwd = Path(str(data.get("cwd", ".")))
        if base_dir is not None:
            if not baseline_config.is_absolute():
                baseline_config = (base_dir / baseline_config).resolve()
            if not cwd.is_absolute():
                cwd = (base_dir / cwd).resolve()
        return cls(
            baseline_config=baseline_config,
            command=tuple(str(item) for item in raw_command),
            cwd=cwd,
            env={str(key): str(value) for key, value in data.get("env", {}).items()},
            timeout_seconds=float(data.get("timeout_seconds", 30.0)),
            classification=ClassificationPolicy.from_dict(data.get("classification")),
            provenance_patterns=tuple(
                str(item) for item in data.get("provenance_patterns", ())
            ),
            roles=tuple(str(item) for item in data.get("roles", ("satisfying", "violating", "repaired"))),
        )

    @classmethod
    def from_path(cls, path: Path) -> "InterventionExecutionManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("intervention execution manifest must be an object")
        return cls.from_dict(payload, base_dir=path.parent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_config": str(self.baseline_config),
            "command": list(self.command),
            "cwd": str(self.cwd),
            "env": dict(self.env),
            "timeout_seconds": self.timeout_seconds,
            "classification": self.classification.to_dict(),
            "provenance_patterns": list(self.provenance_patterns),
            "roles": list(self.roles),
        }


def run_intervention(
    plan_payload: Mapping[str, Any],
    manifest: InterventionExecutionManifest,
    *,
    candidate_index: int = 0,
) -> list[ProbeSample]:
    intervention = resolve_intervention_payload(
        plan_payload,
        candidate_index=candidate_index,
    )
    cases = intervention.get("cases")
    if not isinstance(cases, Mapping):
        raise ValueError("intervention plan must contain cases")
    intervention_id = str(intervention["intervention_id"])
    edge_id = str(intervention["edge_id"])
    primary_parameter = str(intervention["primary_parameter"])
    provenance = intervention.get("provenance", ())
    baseline = json.loads(manifest.baseline_config.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise ValueError("intervention baseline configuration must be an object")

    samples: list[ProbeSample] = []
    with tempfile.TemporaryDirectory(prefix="configfuzz-intervention-") as raw_temp:
        temp_dir = Path(raw_temp)
        for role in manifest.roles:
            case = cases.get(role)
            if not isinstance(case, Mapping) or case.get("status") != "sat":
                continue
            raw_configuration = case.get("configuration")
            if not isinstance(raw_configuration, Mapping):
                raise ValueError(f"intervention case {role!r} has no configuration")
            configuration = copy.deepcopy(baseline)
            resolved_paths = apply_configuration_updates(configuration, raw_configuration)
            config_path = temp_dir / f"{role}.json"
            config_path.write_text(
                json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            template = case.get("probe_sample_template")
            if not isinstance(template, Mapping):
                template = _probe_template(
                    primary_parameter,
                    raw_configuration,
                    intervention_id,
                    edge_id,
                    role,
                )
            substitutions = {
                "config": str(config_path),
                "role": role,
                "intervention_id": intervention_id,
                "edge_id": edge_id,
                "primary_parameter": str(template["parameter"]),
                "primary_value": json.dumps(template["value"], ensure_ascii=False),
                "assignments": json.dumps(raw_configuration, ensure_ascii=False),
                "tracked_parameters": json.dumps(
                    list(resolved_paths.values()), ensure_ascii=False
                ),
            }
            argv = tuple(token.format_map(substitutions) for token in manifest.command)
            observation = execute_command(
                argv,
                cwd=manifest.cwd,
                env={
                    **manifest.env,
                    "CONFIGFUZZ_INTERVENTION_ID": intervention_id,
                    "CONFIGFUZZ_INTERVENTION_EDGE": edge_id,
                    "CONFIGFUZZ_INTERVENTION_ROLE": role,
                },
                timeout_seconds=manifest.timeout_seconds,
            )
            outcome = classify_observation(observation, manifest.classification)
            provenance_matched = (
                outcome.label is OutcomeLabel.INVALID
                and _matches_provenance(
                    observation.combined_output,
                    manifest.provenance_patterns,
                    provenance,
                    substitutions,
                )
            )
            assignments = template.get("assignments", {})
            if not isinstance(assignments, Mapping):
                raise ValueError("probe sample template assignments must be an object")
            samples.append(
                ProbeSample(
                    parameter=str(template["parameter"]),
                    value=template["value"],
                    observation=observation,
                    outcome=outcome,
                    assignments=tuple(
                        (str(name), value) for name, value in assignments.items()
                    ),
                    intervention_id=intervention_id,
                    intervention_edge_id=edge_id,
                    intervention_role=role,
                    provenance_matched=provenance_matched,
                )
            )
    return samples


def intervention_samples_payload(
    plan_payload: Mapping[str, Any],
    manifest: InterventionExecutionManifest,
    samples: list[ProbeSample],
    *,
    candidate_index: int = 0,
) -> dict[str, Any]:
    intervention = resolve_intervention_payload(
        plan_payload,
        candidate_index=candidate_index,
    )
    return {
        "schema_version": 1,
        "manifest": manifest.to_dict(),
        "intervention": intervention,
        "samples": [sample.to_dict() for sample in samples],
    }


def resolve_intervention_payload(
    payload: Mapping[str, Any],
    *,
    candidate_index: int = 0,
) -> Mapping[str, Any]:
    nested = payload.get("intervention")
    if isinstance(nested, Mapping):
        return nested
    selection = payload.get("selection")
    if isinstance(selection, Mapping):
        candidates = selection.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("intervention selection must contain a candidates array")
        if candidate_index < 0 or candidate_index >= len(candidates):
            raise IndexError(
                f"intervention candidate index {candidate_index} is out of range"
            )
        candidate = candidates[candidate_index]
        if not isinstance(candidate, Mapping):
            raise ValueError("selected intervention candidate must be an object")
        intervention = candidate.get("intervention")
        if not isinstance(intervention, Mapping):
            raise ValueError("selected candidate has no intervention plan")
        return intervention
    if isinstance(payload.get("cases"), Mapping):
        return payload
    raise ValueError("input contains neither an intervention nor a selection queue")


def apply_configuration_updates(
    configuration: dict[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name, value in updates.items():
        path = resolve_configuration_path(configuration, str(name))
        _set_path(configuration, path, value)
        resolved[str(name)] = path
    return resolved


def _probe_template(
    primary_parameter: str,
    configuration: Mapping[str, Any],
    intervention_id: str,
    edge_id: str,
    role: str,
) -> dict[str, Any]:
    if primary_parameter not in configuration:
        raise ValueError("primary intervention parameter is absent from the case")
    return {
        "parameter": primary_parameter,
        "value": configuration[primary_parameter],
        "assignments": {
            name: value
            for name, value in configuration.items()
            if name != primary_parameter
        },
        "intervention_id": intervention_id,
        "intervention_edge_id": edge_id,
        "intervention_role": role,
    }


def resolve_configuration_path(configuration: Mapping[str, Any], name: str) -> str:
    if "." in name and _path_exists(configuration, name):
        return name
    leaf = name.rsplit(".", 1)[-1]
    matches = _find_leaf_paths(configuration, leaf)
    if not matches:
        raise KeyError(f"configuration field {name!r} was not found")
    if len(matches) > 1:
        raise ValueError(
            f"configuration field {name!r} is ambiguous: {', '.join(matches)}"
        )
    return matches[0]


def get_configuration_value(configuration: Mapping[str, Any], path: str) -> Any:
    current: Any = configuration
    for part in path.split("."):
        if not isinstance(current, Mapping):
            raise TypeError(f"configuration path {path!r} crosses a non-object field")
        current = current[part]
    return current


def _find_leaf_paths(configuration: Mapping[str, Any], leaf: str) -> list[str]:
    matches: list[str] = []

    def visit(value: Mapping[str, Any], prefix: str = "") -> None:
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key == leaf:
                matches.append(path)
            if isinstance(item, Mapping):
                visit(item, path)

    visit(configuration)
    return matches


def _path_exists(configuration: Mapping[str, Any], path: str) -> bool:
    current: Any = configuration
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _set_path(configuration: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = configuration
    for part in parts[:-1]:
        child = current[part]
        if not isinstance(child, dict):
            raise TypeError(f"configuration path {path!r} crosses a non-object field")
        current = child
    current[parts[-1]] = value


def _matches_provenance(
    output: str,
    configured_patterns: tuple[str, ...],
    provenance: Any,
    substitutions: Mapping[str, str],
) -> bool:
    patterns = [pattern.format_map(substitutions) for pattern in configured_patterns]
    if not patterns and isinstance(provenance, list):
        for item in provenance:
            if not isinstance(item, Mapping) or item.get("source") is None:
                continue
            source = str(item["source"])
            patterns.extend((re.escape(source), re.escape(Path(source).name)))
    return any(re.search(pattern, output, re.IGNORECASE | re.MULTILINE) for pattern in patterns)
