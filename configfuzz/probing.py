from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from configfuzz.outcomes import (
    ClassificationPolicy,
    ClassifiedOutcome,
    ProcessObservation,
    classify_observation,
)


@dataclass(frozen=True, slots=True)
class ProbeManifest:
    parameter: str
    baseline: Any
    command: tuple[str, ...]
    parameter_type: str | None = None
    values: tuple[Any, ...] = ()
    seed_values: tuple[Any, ...] = ()
    enum_values: tuple[Any, ...] = ()
    timeout_seconds: float = 30.0
    cwd: Path = Path(".")
    env: Mapping[str, str] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    classification: ClassificationPolicy = field(default_factory=ClassificationPolicy)
    max_candidates: int = 32

    def __post_init__(self) -> None:
        if not self.parameter.strip():
            raise ValueError("parameter must not be empty")
        if not self.command:
            raise ValueError("command must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "ProbeManifest":
        if "parameter" not in data or "baseline" not in data or "command" not in data:
            raise ValueError("manifest requires parameter, baseline, and command")
        command = data["command"]
        if not isinstance(command, list) or not command:
            raise ValueError("manifest command must be a non-empty JSON array")
        raw_cwd = Path(str(data.get("cwd", ".")))
        if base_dir is not None and not raw_cwd.is_absolute():
            raw_cwd = (base_dir / raw_cwd).resolve()
        return cls(
            parameter=str(data["parameter"]),
            baseline=data["baseline"],
            command=tuple(str(item) for item in command),
            parameter_type=(
                str(data["parameter_type"]).lower()
                if data.get("parameter_type") is not None
                else None
            ),
            values=tuple(data.get("values", ())),
            seed_values=tuple(data.get("seed_values", ())),
            enum_values=tuple(data.get("enum_values", ())),
            timeout_seconds=float(data.get("timeout_seconds", 30.0)),
            cwd=raw_cwd,
            env={str(key): str(value) for key, value in data.get("env", {}).items()},
            context={str(key): value for key, value in data.get("context", {}).items()},
            classification=ClassificationPolicy.from_dict(data.get("classification")),
            max_candidates=int(data.get("max_candidates", 32)),
        )

    @classmethod
    def from_path(cls, path: Path) -> "ProbeManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("manifest root must be a JSON object")
        return cls.from_dict(data, base_dir=path.parent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "baseline": self.baseline,
            "parameter_type": self.parameter_type,
            "command": list(self.command),
            "values": list(self.values),
            "seed_values": list(self.seed_values),
            "enum_values": list(self.enum_values),
            "timeout_seconds": self.timeout_seconds,
            "cwd": str(self.cwd),
            "env": dict(self.env),
            "context": dict(self.context),
            "classification": self.classification.to_dict(),
            "max_candidates": self.max_candidates,
        }


@dataclass(frozen=True, slots=True)
class ProbeSample:
    parameter: str
    value: Any
    observation: ProcessObservation
    outcome: ClassifiedOutcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "value": self.value,
            "observation": self.observation.to_dict(),
            "outcome": self.outcome.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbeSample":
        return cls(
            parameter=str(data["parameter"]),
            value=data["value"],
            observation=ProcessObservation.from_dict(data["observation"]),
            outcome=ClassifiedOutcome.from_dict(data["outcome"]),
        )


def generate_probe_values(manifest: ProbeManifest) -> list[Any]:
    if manifest.values:
        return _deduplicate(manifest.values)[: manifest.max_candidates]

    parameter_type = manifest.parameter_type or _infer_type(manifest.baseline)
    values: list[Any] = [manifest.baseline, *manifest.seed_values]

    if parameter_type in {"int", "integer"}:
        baseline = int(manifest.baseline)
        values.extend((-1, 0, 1, baseline - 1, baseline + 1))
        for raw_context in manifest.context.values():
            if not isinstance(raw_context, int) or isinstance(raw_context, bool):
                continue
            context_value = int(raw_context)
            values.extend((context_value - 1, context_value, context_value + 1))
            if context_value > 0:
                values.extend(
                    divisor
                    for divisor in range(1, min(context_value, 256) + 1)
                    if context_value % divisor == 0
                )
        for power in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            values.extend((power - 1, power, power + 1))
        for seed in manifest.seed_values:
            if isinstance(seed, int) and not isinstance(seed, bool):
                values.extend((seed - 1, seed + 1))
    elif parameter_type in {"float", "number"}:
        baseline = float(manifest.baseline)
        values.extend((-1.0, 0.0, 0.5, 1.0, baseline - 1.0, baseline + 1.0))
        for seed in manifest.seed_values:
            if isinstance(seed, (int, float)) and not isinstance(seed, bool):
                values.extend((float(seed) - 1e-6, float(seed) + 1e-6))
    elif parameter_type in {"bool", "boolean"}:
        values.extend((False, True))
    elif parameter_type in {"enum", "string", "str"}:
        values.extend(manifest.enum_values)
        if parameter_type in {"string", "str"}:
            values.extend(("", "__configfuzz_unknown__"))
    else:
        raise ValueError(f"unsupported parameter_type: {parameter_type!r}")

    return _deduplicate(values)[: manifest.max_candidates]


def run_probe(manifest: ProbeManifest, value: Any) -> ProbeSample:
    substitutions = {
        "parameter": manifest.parameter,
        "value": _format_value(value),
    }
    argv = tuple(token.format_map(substitutions) for token in manifest.command)
    env = os.environ.copy()
    env.update(manifest.env)
    env["CONFIGFUZZ_PARAMETER"] = manifest.parameter
    env["CONFIGFUZZ_VALUE"] = _format_value(value)

    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=manifest.cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=manifest.timeout_seconds,
            check=False,
        )
        observation = ProcessObservation(
            argv=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        observation = ProcessObservation(
            argv=argv,
            returncode=None,
            stdout=_coerce_timeout_text(exc.stdout),
            stderr=_coerce_timeout_text(exc.stderr),
            duration_seconds=time.monotonic() - started,
            timed_out=True,
        )

    return ProbeSample(
        parameter=manifest.parameter,
        value=value,
        observation=observation,
        outcome=classify_observation(observation, manifest.classification),
    )


def run_manifest(manifest: ProbeManifest) -> list[ProbeSample]:
    return [run_probe(manifest, value) for value in generate_probe_values(manifest)]


def samples_payload(manifest: ProbeManifest, samples: Iterable[ProbeSample]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest": manifest.to_dict(),
        "samples": [sample.to_dict() for sample in samples],
    }


def load_samples(path: Path) -> tuple[dict[str, Any], list[ProbeSample]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("sample file must contain a samples array")
    return payload, [ProbeSample.from_dict(item) for item in payload["samples"]]


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    raise ValueError(f"cannot infer parameter type from {type(value).__name__}")


def _deduplicate(values: Iterable[Any]) -> list[Any]:
    seen: set[tuple[type[Any], str]] = set()
    result: list[Any] = []
    for value in values:
        key = (type(value), repr(value))
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _coerce_timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
