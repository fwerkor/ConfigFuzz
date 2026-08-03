from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Iterable


class ConstraintKind(str, Enum):
    TYPE = "type"
    RANGE = "range"
    ENUM = "enum"
    RELATION = "relation"
    CONDITIONAL = "conditional"
    ENVIRONMENT = "environment"
    RESOURCE = "resource"
    OTHER = "other"


class EvidenceKind(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    DOCUMENTATION = "documentation"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class Evidence:
    kind: EvidenceKind
    source: str
    line: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True, slots=True)
class Constraint:
    expression: str
    kind: ConstraintKind
    parameters: tuple[str, ...]
    evidence: tuple[Evidence, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        expression = self.expression.strip()
        if not expression:
            raise ValueError("constraint expression must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "parameters", tuple(dict.fromkeys(self.parameters)))

    @property
    def key(self) -> tuple[str, str, tuple[str, ...]]:
        return self.expression, self.kind.value, self.parameters

    def merge(self, other: "Constraint") -> "Constraint":
        if self.key != other.key:
            raise ValueError("only equivalent constraints can be merged")
        evidence = tuple(dict.fromkeys((*self.evidence, *other.evidence)))
        return replace(self, evidence=evidence, confidence=max(self.confidence, other.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "kind": self.kind.value,
            "parameters": list(self.parameters),
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(slots=True)
class ConstraintSet:
    parameter: str
    constraints: list[Constraint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, constraint: Constraint) -> None:
        if self.parameter not in constraint.parameters:
            raise ValueError(
                f"constraint {constraint.expression!r} does not reference {self.parameter!r}"
            )
        for index, existing in enumerate(self.constraints):
            if existing.key == constraint.key:
                self.constraints[index] = existing.merge(constraint)
                return
        self.constraints.append(constraint)

    def extend(self, constraints: Iterable[Constraint]) -> None:
        for constraint in constraints:
            self.add(constraint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "constraints": [item.to_dict() for item in self.constraints],
            "metadata": self.metadata,
        }
