from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


_PATTERN_FLAGS = re.IGNORECASE | re.MULTILINE


class OutcomeLabel(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    POTENTIAL_BUG = "potential_bug"


@dataclass(frozen=True, slots=True)
class ProcessObservation:
    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessObservation":
        return cls(
            argv=tuple(str(item) for item in data.get("argv", [])),
            returncode=data.get("returncode"),
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            timed_out=bool(data.get("timed_out", False)),
        )


@dataclass(frozen=True, slots=True)
class ClassificationPolicy:
    invalid_patterns: tuple[str, ...] = ()
    infrastructure_patterns: tuple[str, ...] = ()
    bug_patterns: tuple[str, ...] = ()
    milestone_patterns: tuple[str, ...] = ()
    unexpected_failure_after_milestone_is_bug: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ClassificationPolicy":
        data = data or {}
        return cls(
            invalid_patterns=_strings(data.get("invalid_patterns", ())),
            infrastructure_patterns=_strings(data.get("infrastructure_patterns", ())),
            bug_patterns=_strings(data.get("bug_patterns", ())),
            milestone_patterns=_strings(data.get("milestone_patterns", ())),
            unexpected_failure_after_milestone_is_bug=bool(
                data.get("unexpected_failure_after_milestone_is_bug", True)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "invalid_patterns": list(self.invalid_patterns),
            "infrastructure_patterns": list(self.infrastructure_patterns),
            "bug_patterns": list(self.bug_patterns),
            "milestone_patterns": list(self.milestone_patterns),
            "unexpected_failure_after_milestone_is_bug": (
                self.unexpected_failure_after_milestone_is_bug
            ),
        }


@dataclass(frozen=True, slots=True)
class ClassifiedOutcome:
    label: OutcomeLabel
    reason: str
    matched_pattern: str | None = None
    reached_milestone: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["label"] = self.label.value
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassifiedOutcome":
        return cls(
            label=OutcomeLabel(data["label"]),
            reason=str(data.get("reason", "")),
            matched_pattern=data.get("matched_pattern"),
            reached_milestone=bool(data.get("reached_milestone", False)),
        )


def classify_observation(
    observation: ProcessObservation,
    policy: ClassificationPolicy,
) -> ClassifiedOutcome:
    output = observation.combined_output
    reached_milestone, milestone_pattern = _matches_any(
        output, policy.milestone_patterns
    )

    matched, pattern = _matches_any(output, policy.infrastructure_patterns)
    if matched:
        return ClassifiedOutcome(
            OutcomeLabel.UNKNOWN,
            "matched an infrastructure-failure pattern",
            pattern,
            reached_milestone,
        )

    matched, pattern = _matches_any(output, policy.invalid_patterns)
    if matched:
        return ClassifiedOutcome(
            OutcomeLabel.INVALID,
            "matched an explicit configuration-rejection pattern",
            pattern,
            reached_milestone,
        )

    matched, pattern = _matches_any(output, policy.bug_patterns)
    if matched:
        return ClassifiedOutcome(
            OutcomeLabel.POTENTIAL_BUG,
            "matched an explicit bug-oracle pattern",
            pattern,
            reached_milestone,
        )

    if observation.timed_out:
        return ClassifiedOutcome(
            OutcomeLabel.UNKNOWN,
            "execution timed out without a conclusive oracle",
            reached_milestone=reached_milestone,
        )

    if observation.returncode == 0:
        if policy.milestone_patterns and not reached_milestone:
            return ClassifiedOutcome(
                OutcomeLabel.UNKNOWN,
                "process exited successfully but did not reach the configured milestone",
                reached_milestone=False,
            )
        reason = "process completed and reached the configured milestone"
        if not policy.milestone_patterns:
            reason = "process completed successfully"
        return ClassifiedOutcome(
            OutcomeLabel.VALID,
            reason,
            milestone_pattern,
            reached_milestone or not policy.milestone_patterns,
        )

    if reached_milestone and policy.unexpected_failure_after_milestone_is_bug:
        return ClassifiedOutcome(
            OutcomeLabel.POTENTIAL_BUG,
            "process failed after passing the configured validation milestone",
            milestone_pattern,
            True,
        )

    return ClassifiedOutcome(
        OutcomeLabel.UNKNOWN,
        "non-zero exit was not attributable to configuration validation or a bug oracle",
        reached_milestone=reached_milestone,
    )


def _matches_any(text: str, patterns: Iterable[str]) -> tuple[bool, str | None]:
    for pattern in patterns:
        if re.search(pattern, text, _PATTERN_FLAGS):
            return True, pattern
    return False, None


def _strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)
