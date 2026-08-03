from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from configfuzz.model import ConstraintKind


class Enforcement(str, Enum):
    REJECT = "reject"
    REPAIR = "repair"
    WARN = "warn"
    SAMPLE = "sample"
    DEFAULT = "default"


class RuleStrength(str, Enum):
    FRAMEWORK_REQUIREMENT = "framework_requirement"
    LMSV_POLICY = "lmsv_policy"
    ENVIRONMENT_LIMIT = "environment_limit"
    RESOURCE_LIMIT = "resource_limit"
    WORKAROUND = "workaround"
    EMPIRICAL = "empirical"
    UNKNOWN = "unknown"


class RuleStatus(str, Enum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    VALIDATED = "validated"
    CONTRADICTED = "contradicted"
    DEPRECATED = "deprecated"


@dataclass(frozen=True, slots=True)
class RuleSource:
    file: str
    lines: tuple[int, int] | None = None
    symbol: str | None = None
    source_type: str = "implementation"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleSource":
        raw_lines = data.get("lines")
        lines = tuple(raw_lines) if raw_lines is not None else None
        if lines is not None and (len(lines) != 2 or lines[0] < 1 or lines[1] < lines[0]):
            raise ValueError(f"invalid source line range: {raw_lines!r}")
        return cls(
            file=str(data["file"]),
            lines=lines,
            symbol=data.get("symbol"),
            source_type=str(data.get("source_type", "implementation")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"file": self.file, "source_type": self.source_type}
        if self.lines is not None:
            data["lines"] = list(self.lines)
        if self.symbol is not None:
            data["symbol"] = self.symbol
        return data


@dataclass(frozen=True, slots=True)
class RuleScope:
    tasks: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()
    hardware: tuple[str, ...] = ()
    stage: str | None = None
    condition: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RuleScope":
        data = data or {}
        return cls(
            tasks=tuple(str(item) for item in data.get("tasks", [])),
            models=tuple(str(item) for item in data.get("models", [])),
            backends=tuple(str(item) for item in data.get("backends", [])),
            frameworks=tuple(str(item) for item in data.get("frameworks", [])),
            hardware=tuple(str(item) for item in data.get("hardware", [])),
            stage=data.get("stage"),
            condition=data.get("condition"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key in ("tasks", "models", "backends", "frameworks", "hardware"):
            value = getattr(self, key)
            if value:
                data[key] = list(value)
        if self.stage is not None:
            data["stage"] = self.stage
        if self.condition is not None:
            data["condition"] = self.condition
        return data


@dataclass(frozen=True, slots=True)
class ManualConstraintRule:
    id: str
    expression: str
    kind: ConstraintKind
    parameters: tuple[str, ...]
    enforcement: Enforcement
    strength: RuleStrength
    status: RuleStatus
    scope: RuleScope
    sources: tuple[RuleSource, ...]
    rationale: str
    repair: dict[str, Any] | None = None
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManualConstraintRule":
        rule = cls(
            id=str(data["id"]),
            expression=str(data["expression"]).strip(),
            kind=ConstraintKind(data["kind"]),
            parameters=tuple(str(item) for item in data["parameters"]),
            enforcement=Enforcement(data["enforcement"]),
            strength=RuleStrength(data["strength"]),
            status=RuleStatus(data["status"]),
            scope=RuleScope.from_dict(data.get("scope")),
            sources=tuple(RuleSource.from_dict(item) for item in data["sources"]),
            rationale=str(data.get("rationale", "")).strip(),
            repair=data.get("repair"),
            aliases=tuple(str(item) for item in data.get("aliases", [])),
            metadata=dict(data.get("metadata", {})),
        )
        rule.validate()
        return rule

    def validate(self) -> None:
        if not self.id or any(char.isspace() for char in self.id):
            raise ValueError(f"invalid rule id: {self.id!r}")
        if not self.expression:
            raise ValueError(f"rule {self.id}: empty expression")
        if not self.parameters:
            raise ValueError(f"rule {self.id}: parameters must not be empty")
        if not self.sources:
            raise ValueError(f"rule {self.id}: sources must not be empty")
        if not self.rationale:
            raise ValueError(f"rule {self.id}: rationale must not be empty")
        if self.enforcement is Enforcement.REPAIR and not self.repair:
            raise ValueError(f"rule {self.id}: repair enforcement requires repair metadata")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "expression": self.expression,
            "kind": self.kind.value,
            "parameters": list(self.parameters),
            "enforcement": self.enforcement.value,
            "strength": self.strength.value,
            "status": self.status.value,
            "scope": self.scope.to_dict(),
            "sources": [item.to_dict() for item in self.sources],
            "rationale": self.rationale,
        }
        if self.repair is not None:
            data["repair"] = self.repair
        if self.aliases:
            data["aliases"] = list(self.aliases)
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass(slots=True)
class ConstraintCorpus:
    name: str
    baseline: dict[str, Any]
    rules: list[ManualConstraintRule]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConstraintCorpus":
        corpus = cls(
            name=str(data["name"]),
            baseline=dict(data.get("baseline", {})),
            rules=[ManualConstraintRule.from_dict(item) for item in data.get("rules", [])],
            schema_version=int(data.get("schema_version", 1)),
        )
        corpus.validate()
        return corpus

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported corpus schema version: {self.schema_version}")
        if not self.name:
            raise ValueError("corpus name must not be empty")
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id: {rule.id}")
            seen.add(rule.id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "baseline": self.baseline,
            "rules": [rule.to_dict() for rule in self.rules],
        }


def load_corpus(path: str | Path) -> ConstraintCorpus:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("constraint corpus root must be a mapping")
    return ConstraintCorpus.from_dict(raw)


def dump_corpus(corpus: ConstraintCorpus, path: str | Path) -> None:
    corpus.validate()
    output = yaml.safe_dump(
        corpus.to_dict(),
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    Path(path).write_text(output, encoding="utf-8")
