from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from configfuzz.corpus import (
    ConstraintCorpus,
    ManualConstraintRule,
    RuleStrength,
    load_corpus,
)


class ConstraintCategory(str, Enum):
    LOCAL = "local"
    STRUCTURAL = "structural"
    FEATURE_INTERACTION = "feature_interaction"
    ENVIRONMENT_DEPENDENCY = "environment_dependency"


class SemanticClass(str, Enum):
    MATHEMATICAL_INVARIANT = "mathematical_invariant"
    IMPLEMENTATION_LIMIT = "implementation_limit"
    RESOURCE_RISK = "resource_risk"
    OPTIMIZATION_PREFERENCE = "optimization_preference"
    UNKNOWN = "unknown"


class SoftwareLayer(str, Enum):
    ARGUMENT_PARSING = "argument_parsing"
    VALIDATOR = "validator"
    MODEL_CONSTRUCTION = "model_construction"
    PARALLEL = "parallel"
    COMMUNICATION = "communication"
    KERNEL = "kernel"
    TRAINING = "training"
    CHECKPOINT = "checkpoint"
    OTHER = "other"


class ValidationCoverage(str, Enum):
    FULL_EXPLICIT = "full_explicit"
    PARTIAL = "partial"
    IMPLICIT_DELAYED = "implicit_delayed"
    UNCOVERED = "uncovered"
    UNREVIEWED = "unreviewed"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    PRIMARY_REVIEWED = "primary_reviewed"
    SECONDARY_REVIEWED = "secondary_reviewed"
    ADJUDICATED = "adjudicated"


class ExecutionMilestone(str, Enum):
    CONFIG_GENERATION = "config_generation"
    ARGUMENT_PARSING = "argument_parsing"
    CONFIG_VALIDATION = "config_validation"
    MODEL_CONSTRUCTION = "model_construction"
    PROCESS_GROUP_INITIALIZATION = "process_group_initialization"
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER_STEP = "optimizer_step"
    REPEATED_TRAINING = "repeated_training"
    CHECKPOINT_SAVE_LOAD = "checkpoint_save_load"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


MILESTONE_ORDER: dict[ExecutionMilestone, int] = {
    ExecutionMilestone.CONFIG_GENERATION: 0,
    ExecutionMilestone.ARGUMENT_PARSING: 1,
    ExecutionMilestone.CONFIG_VALIDATION: 2,
    ExecutionMilestone.MODEL_CONSTRUCTION: 3,
    ExecutionMilestone.PROCESS_GROUP_INITIALIZATION: 4,
    ExecutionMilestone.FORWARD: 5,
    ExecutionMilestone.BACKWARD: 6,
    ExecutionMilestone.OPTIMIZER_STEP: 7,
    ExecutionMilestone.REPEATED_TRAINING: 8,
    ExecutionMilestone.CHECKPOINT_SAVE_LOAD: 9,
    ExecutionMilestone.COMPLETED: 10,
    ExecutionMilestone.TIMEOUT: -1,
    ExecutionMilestone.UNKNOWN: -1,
}


class ExperimentMethod(str, Enum):
    RAW_MUTATION = "raw_mutation"
    NATIVE_VALIDATOR_GUIDED = "native_validator_guided"
    CONSTRAINT_FILTER_ONLY = "constraint_filter_only"
    STATIC_HARD_CONFIGFUZZ = "static_hard_configfuzz"
    CONFIGFUZZ = "configfuzz"
    GLOBAL_REPAIR = "global_repair"


class ExperimentOutcome(str, Enum):
    VALID = "valid"
    EXPECTED_REJECTION = "expected_rejection"
    RESOURCE_FAILURE = "resource_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    UNEXPLAINED_FAILURE = "unexplained_failure"
    POTENTIAL_BUG = "potential_bug"
    UNKNOWN = "unknown"


class ConstraintPairRole(str, Enum):
    SATISFYING = "satisfying"
    VIOLATING = "violating"


class FailureMode(str, Enum):
    NONE = "none"
    EXPLICIT_REJECTION = "explicit_rejection"
    CRASH = "crash"
    HANG = "hang"
    SILENT_ABNORMAL = "silent_abnormal"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class BugStatus(str, Enum):
    HISTORICAL_REPLAY = "historical_replay"
    DEVELOPER_CONFIRMED = "developer_confirmed"
    FIXED = "fixed"
    ACCEPTED_PENDING = "accepted_pending"
    REJECTED = "rejected"
    UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True, slots=True)
class CoverageEvidence:
    file: str
    lines: tuple[int, int] | None = None
    symbol: str | None = None
    detail: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CoverageEvidence":
        raw_lines = data.get("lines")
        lines = (
            tuple(int(item) for item in raw_lines) if raw_lines is not None else None
        )
        if lines is not None and (
            len(lines) != 2 or lines[0] < 1 or lines[1] < lines[0]
        ):
            raise ValueError(f"invalid coverage-evidence line range: {raw_lines!r}")
        return cls(
            file=_required_string(data, "file"),
            lines=lines,
            symbol=_optional_string(data.get("symbol")),
            detail=_optional_string(data.get("detail")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"file": self.file}
        if self.lines is not None:
            data["lines"] = list(self.lines)
        if self.symbol is not None:
            data["symbol"] = self.symbol
        if self.detail is not None:
            data["detail"] = self.detail
        return data


@dataclass(frozen=True, slots=True)
class ConstraintAuditRecord:
    constraint_id: str
    participants: tuple[str, ...]
    predicate: str
    guard: str | None
    scope: Mapping[str, Any]
    arity: int
    category: ConstraintCategory
    semantic_class: SemanticClass
    provenance: tuple[Mapping[str, Any], ...]
    software_layers: tuple[SoftwareLayer, ...]
    first_affected_milestone: ExecutionMilestone
    native_validation: ValidationCoverage
    coverage_evidence: tuple[CoverageEvidence, ...] = ()
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewers: tuple[str, ...] = ()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.constraint_id or any(char.isspace() for char in self.constraint_id):
            raise ValueError(f"invalid constraint id: {self.constraint_id!r}")
        if not self.participants:
            raise ValueError(
                f"constraint {self.constraint_id}: participants must not be empty"
            )
        if self.arity != len(self.participants):
            raise ValueError(
                f"constraint {self.constraint_id}: arity {self.arity} does not match "
                f"{len(self.participants)} participants"
            )
        if not self.predicate.strip():
            raise ValueError(
                f"constraint {self.constraint_id}: predicate must not be empty"
            )
        if not self.software_layers:
            raise ValueError(
                f"constraint {self.constraint_id}: software_layers must not be empty"
            )
        if (
            self.native_validation is not ValidationCoverage.UNREVIEWED
            and not self.coverage_evidence
        ):
            raise ValueError(
                f"constraint {self.constraint_id}: reviewed native validation requires evidence"
            )
        if self.review_status is not ReviewStatus.UNREVIEWED and not self.reviewers:
            raise ValueError(
                f"constraint {self.constraint_id}: reviewed record requires reviewer identity"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConstraintAuditRecord":
        participants = tuple(str(item) for item in data.get("participants", ()))
        return cls(
            constraint_id=_required_string(data, "constraint_id"),
            participants=participants,
            predicate=_required_string(data, "predicate"),
            guard=_optional_string(data.get("guard")),
            scope=_mapping(data.get("scope", {}), "scope"),
            arity=int(data.get("arity", len(participants))),
            category=ConstraintCategory(str(data["category"])),
            semantic_class=SemanticClass(str(data["semantic_class"])),
            provenance=tuple(
                _mapping(item, "provenance entry")
                for item in data.get("provenance", ())
            ),
            software_layers=tuple(
                SoftwareLayer(str(item)) for item in data.get("software_layers", ())
            ),
            first_affected_milestone=ExecutionMilestone(
                str(data.get("first_affected_milestone", "unknown"))
            ),
            native_validation=ValidationCoverage(
                str(data.get("native_validation", "unreviewed"))
            ),
            coverage_evidence=tuple(
                CoverageEvidence.from_dict(_mapping(item, "coverage evidence"))
                for item in data.get("coverage_evidence", ())
            ),
            review_status=ReviewStatus(str(data.get("review_status", "unreviewed"))),
            reviewers=tuple(str(item) for item in data.get("reviewers", ())),
            notes=_optional_string(data.get("notes")),
            metadata=_mapping(data.get("metadata", {}), "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "constraint_id": self.constraint_id,
            "participants": list(self.participants),
            "predicate": self.predicate,
            "guard": self.guard,
            "scope": dict(self.scope),
            "arity": self.arity,
            "category": self.category.value,
            "semantic_class": self.semantic_class.value,
            "provenance": [dict(item) for item in self.provenance],
            "software_layers": [item.value for item in self.software_layers],
            "first_affected_milestone": self.first_affected_milestone.value,
            "native_validation": self.native_validation.value,
            "coverage_evidence": [item.to_dict() for item in self.coverage_evidence],
            "review_status": self.review_status.value,
            "reviewers": list(self.reviewers),
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }
        return data


@dataclass(slots=True)
class ConstraintAuditDataset:
    name: str
    source_corpus: Mapping[str, Any]
    records: list[ConstraintAuditRecord]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConstraintAuditDataset":
        dataset = cls(
            name=_required_string(data, "name"),
            source_corpus=_mapping(data.get("source_corpus", {}), "source_corpus"),
            records=[
                ConstraintAuditRecord.from_dict(_mapping(item, "constraint record"))
                for item in data.get("records", ())
            ],
            schema_version=int(data.get("schema_version", 1)),
        )
        dataset.validate()
        return dataset

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported constraint-audit schema: {self.schema_version}"
            )
        seen: set[str] = set()
        for record in self.records:
            if record.constraint_id in seen:
                raise ValueError(f"duplicate constraint id: {record.constraint_id}")
            seen.add(record.constraint_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "source_corpus": dict(self.source_corpus),
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class MutationIntent:
    intent_id: str
    workload_id: str
    baseline_id: str
    target_parameter: str
    target_value: Any
    intent_class: str
    intent_pool: str = "method_independent"
    source_constraint_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MutationIntent":
        return cls(
            intent_id=_required_string(data, "intent_id"),
            workload_id=_required_string(data, "workload_id"),
            baseline_id=_required_string(data, "baseline_id"),
            target_parameter=_required_string(data, "target_parameter"),
            target_value=data.get("target_value"),
            intent_class=_required_string(data, "intent_class"),
            intent_pool=str(data.get("intent_pool", "method_independent")),
            source_constraint_ids=tuple(
                str(item) for item in data.get("source_constraint_ids", ())
            ),
            metadata=_mapping(data.get("metadata", {}), "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "workload_id": self.workload_id,
            "baseline_id": self.baseline_id,
            "target_parameter": self.target_parameter,
            "target_value": self.target_value,
            "intent_class": self.intent_class,
            "intent_pool": self.intent_pool,
            "source_constraint_ids": list(self.source_constraint_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExperimentRunRecord:
    run_id: str
    rq: str
    method: ExperimentMethod
    workload_id: str
    baseline_id: str
    intent_id: str | None
    seed: int | None
    generated: bool
    target_value_preserved: bool | None
    coordinated_parameters: tuple[str, ...]
    modification_distance: float | None
    solver_seconds: float
    deepest_milestone: ExecutionMilestone
    outcome: ExperimentOutcome
    duration_seconds: float
    gpu_seconds: float
    peak_memory_mib: float | None
    timed_out: bool
    intent_pool: str | None = None
    constraint_id: str | None = None
    pair_role: ConstraintPairRole | None = None
    first_failure_milestone: ExecutionMilestone | None = None
    failure_mode: FailureMode | None = None
    error_message_interpretable: bool | None = None
    campaign_test_index: int | None = None
    campaign_elapsed_seconds: float | None = None
    campaign_gpu_seconds: float | None = None
    constraints_exercised: tuple[str, ...] = ()
    boundaries_exercised: tuple[str, ...] = ()
    guard_transitions: tuple[str, ...] = ()
    topologies: tuple[str, ...] = ()
    feature_interactions: tuple[str, ...] = ()
    backend_paths: tuple[str, ...] = ()
    behavior_ids: tuple[str, ...] = ()
    behavior_signature: str | None = None
    active_constraint_ids: tuple[str, ...] = ()
    affected_region: tuple[str, ...] = ()
    solver_modifications: Mapping[str, Any] = field(default_factory=dict)
    constraint_status_before: Mapping[str, str] = field(default_factory=dict)
    constraint_status_after: Mapping[str, str] = field(default_factory=dict)
    refined_constraint_ids: tuple[str, ...] = ()
    constraint_provenance_ids: tuple[str, ...] = ()
    bug_id: str | None = None
    root_cause_id: str | None = None
    buggy_failed: bool | None = None
    fixed_passed: bool | None = None
    root_cause_match: bool | None = None
    bug_status: BugStatus | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.rq not in {"rq1", "rq2", "rq3"}:
            raise ValueError(f"unsupported research question: {self.rq!r}")
        for name, value in (
            ("solver_seconds", self.solver_seconds),
            ("duration_seconds", self.duration_seconds),
            ("gpu_seconds", self.gpu_seconds),
        ):
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.modification_distance is not None and (
            self.modification_distance < 0
            or not math.isfinite(self.modification_distance)
        ):
            raise ValueError("modification_distance must be finite and non-negative")
        if self.pair_role is not None and self.constraint_id is None:
            raise ValueError("constraint pair role requires constraint_id")
        if self.campaign_test_index is not None and self.campaign_test_index < 1:
            raise ValueError("campaign_test_index must be positive")
        for name, value in (
            ("campaign_elapsed_seconds", self.campaign_elapsed_seconds),
            ("campaign_gpu_seconds", self.campaign_gpu_seconds),
        ):
            if value is not None and (value < 0 or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentRunRecord":
        return cls(
            run_id=_required_string(data, "run_id"),
            rq=_required_string(data, "rq").lower(),
            method=ExperimentMethod(str(data["method"])),
            workload_id=_required_string(data, "workload_id"),
            baseline_id=_required_string(data, "baseline_id"),
            intent_id=_optional_string(data.get("intent_id")),
            intent_pool=_optional_string(data.get("intent_pool")),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            generated=bool(data.get("generated", False)),
            target_value_preserved=(
                bool(data["target_value_preserved"])
                if data.get("target_value_preserved") is not None
                else None
            ),
            coordinated_parameters=tuple(
                str(item) for item in data.get("coordinated_parameters", ())
            ),
            modification_distance=(
                float(data["modification_distance"])
                if data.get("modification_distance") is not None
                else None
            ),
            solver_seconds=float(data.get("solver_seconds", 0.0)),
            deepest_milestone=ExecutionMilestone(
                str(data.get("deepest_milestone", "unknown"))
            ),
            outcome=ExperimentOutcome(str(data.get("outcome", "unknown"))),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            gpu_seconds=float(data.get("gpu_seconds", 0.0)),
            peak_memory_mib=(
                float(data["peak_memory_mib"])
                if data.get("peak_memory_mib") is not None
                else None
            ),
            timed_out=bool(data.get("timed_out", False)),
            constraint_id=_optional_string(data.get("constraint_id")),
            pair_role=(
                ConstraintPairRole(str(data["pair_role"]))
                if data.get("pair_role") is not None
                else None
            ),
            first_failure_milestone=(
                ExecutionMilestone(str(data["first_failure_milestone"]))
                if data.get("first_failure_milestone") is not None
                else None
            ),
            failure_mode=(
                FailureMode(str(data["failure_mode"]))
                if data.get("failure_mode") is not None
                else None
            ),
            error_message_interpretable=(
                bool(data["error_message_interpretable"])
                if data.get("error_message_interpretable") is not None
                else None
            ),
            campaign_test_index=(
                int(data["campaign_test_index"])
                if data.get("campaign_test_index") is not None
                else None
            ),
            campaign_elapsed_seconds=(
                float(data["campaign_elapsed_seconds"])
                if data.get("campaign_elapsed_seconds") is not None
                else None
            ),
            campaign_gpu_seconds=(
                float(data["campaign_gpu_seconds"])
                if data.get("campaign_gpu_seconds") is not None
                else None
            ),
            constraints_exercised=_string_tuple(data.get("constraints_exercised", ())),
            boundaries_exercised=_string_tuple(data.get("boundaries_exercised", ())),
            guard_transitions=_string_tuple(data.get("guard_transitions", ())),
            topologies=_string_tuple(data.get("topologies", ())),
            feature_interactions=_string_tuple(data.get("feature_interactions", ())),
            backend_paths=_string_tuple(data.get("backend_paths", ())),
            behavior_ids=_string_tuple(data.get("behavior_ids", ())),
            behavior_signature=_optional_string(data.get("behavior_signature")),
            active_constraint_ids=_string_tuple(data.get("active_constraint_ids", ())),
            affected_region=_string_tuple(data.get("affected_region", ())),
            solver_modifications=_mapping(
                data.get("solver_modifications", {}), "solver_modifications"
            ),
            constraint_status_before={
                str(key): str(value)
                for key, value in _mapping(
                    data.get("constraint_status_before", {}),
                    "constraint_status_before",
                ).items()
            },
            constraint_status_after={
                str(key): str(value)
                for key, value in _mapping(
                    data.get("constraint_status_after", {}),
                    "constraint_status_after",
                ).items()
            },
            refined_constraint_ids=_string_tuple(
                data.get("refined_constraint_ids", ())
            ),
            constraint_provenance_ids=_string_tuple(
                data.get("constraint_provenance_ids", ())
            ),
            bug_id=_optional_string(data.get("bug_id")),
            root_cause_id=_optional_string(data.get("root_cause_id")),
            buggy_failed=(
                bool(data["buggy_failed"])
                if data.get("buggy_failed") is not None
                else None
            ),
            fixed_passed=(
                bool(data["fixed_passed"])
                if data.get("fixed_passed") is not None
                else None
            ),
            root_cause_match=(
                bool(data["root_cause_match"])
                if data.get("root_cause_match") is not None
                else None
            ),
            bug_status=(
                BugStatus(str(data["bug_status"]))
                if data.get("bug_status") is not None
                else None
            ),
            metadata=_mapping(data.get("metadata", {}), "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "rq": self.rq,
            "method": self.method.value,
            "workload_id": self.workload_id,
            "baseline_id": self.baseline_id,
            "intent_id": self.intent_id,
            "intent_pool": self.intent_pool,
            "seed": self.seed,
            "generated": self.generated,
            "target_value_preserved": self.target_value_preserved,
            "coordinated_parameters": list(self.coordinated_parameters),
            "modification_distance": self.modification_distance,
            "solver_seconds": self.solver_seconds,
            "deepest_milestone": self.deepest_milestone.value,
            "outcome": self.outcome.value,
            "duration_seconds": self.duration_seconds,
            "gpu_seconds": self.gpu_seconds,
            "peak_memory_mib": self.peak_memory_mib,
            "timed_out": self.timed_out,
            "constraint_id": self.constraint_id,
            "pair_role": self.pair_role.value if self.pair_role is not None else None,
            "first_failure_milestone": (
                self.first_failure_milestone.value
                if self.first_failure_milestone is not None
                else None
            ),
            "failure_mode": (
                self.failure_mode.value if self.failure_mode is not None else None
            ),
            "error_message_interpretable": self.error_message_interpretable,
            "campaign_test_index": self.campaign_test_index,
            "campaign_elapsed_seconds": self.campaign_elapsed_seconds,
            "campaign_gpu_seconds": self.campaign_gpu_seconds,
            "constraints_exercised": list(self.constraints_exercised),
            "boundaries_exercised": list(self.boundaries_exercised),
            "guard_transitions": list(self.guard_transitions),
            "topologies": list(self.topologies),
            "feature_interactions": list(self.feature_interactions),
            "backend_paths": list(self.backend_paths),
            "behavior_ids": list(self.behavior_ids),
            "behavior_signature": self.behavior_signature,
            "active_constraint_ids": list(self.active_constraint_ids),
            "affected_region": list(self.affected_region),
            "solver_modifications": dict(self.solver_modifications),
            "constraint_status_before": dict(self.constraint_status_before),
            "constraint_status_after": dict(self.constraint_status_after),
            "refined_constraint_ids": list(self.refined_constraint_ids),
            "constraint_provenance_ids": list(self.constraint_provenance_ids),
            "bug_id": self.bug_id,
            "root_cause_id": self.root_cause_id,
            "buggy_failed": self.buggy_failed,
            "fixed_passed": self.fixed_passed,
            "root_cause_match": self.root_cause_match,
            "bug_status": self.bug_status.value
            if self.bug_status is not None
            else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HistoricalBugRecord:
    bug_id: str
    split: str
    issue_or_pr: str
    buggy_commit: str
    fixed_commit: str
    affected_parameters: tuple[str, ...]
    workload_id: str
    environment: Mapping[str, Any]
    failure_milestone: ExecutionMilestone
    failure_signature: str
    root_cause: str
    constraint_type: ConstraintCategory
    arity: int
    oracle: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HistoricalBugRecord":
        parameters = _string_tuple(data.get("affected_parameters", ()))
        split = _required_string(data, "split")
        if split not in {"development", "evaluation"}:
            raise ValueError(f"unsupported historical-bug split: {split!r}")
        record = cls(
            bug_id=_required_string(data, "bug_id"),
            split=split,
            issue_or_pr=_required_string(data, "issue_or_pr"),
            buggy_commit=_required_string(data, "buggy_commit"),
            fixed_commit=_required_string(data, "fixed_commit"),
            affected_parameters=parameters,
            workload_id=_required_string(data, "workload_id"),
            environment=_mapping(data.get("environment", {}), "environment"),
            failure_milestone=ExecutionMilestone(str(data["failure_milestone"])),
            failure_signature=_required_string(data, "failure_signature"),
            root_cause=_required_string(data, "root_cause"),
            constraint_type=ConstraintCategory(str(data["constraint_type"])),
            arity=int(data.get("arity", len(parameters))),
            oracle=_required_string(data, "oracle"),
            metadata=_mapping(data.get("metadata", {}), "metadata"),
        )
        if record.arity != len(record.affected_parameters):
            raise ValueError(
                f"bug {record.bug_id}: arity does not match affected_parameters"
            )
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "bug_id": self.bug_id,
            "split": self.split,
            "issue_or_pr": self.issue_or_pr,
            "buggy_commit": self.buggy_commit,
            "fixed_commit": self.fixed_commit,
            "affected_parameters": list(self.affected_parameters),
            "workload_id": self.workload_id,
            "environment": dict(self.environment),
            "failure_milestone": self.failure_milestone.value,
            "failure_signature": self.failure_signature,
            "root_cause": self.root_cause,
            "constraint_type": self.constraint_type.value,
            "arity": self.arity,
            "oracle": self.oracle,
            "metadata": dict(self.metadata),
        }


def build_rq1_audit_dataset(corpus: ConstraintCorpus) -> ConstraintAuditDataset:
    records = [_audit_record_from_rule(rule) for rule in corpus.rules]
    return ConstraintAuditDataset(
        name=f"{corpus.name}-rq1-audit",
        source_corpus={
            "name": corpus.name,
            "schema_version": corpus.schema_version,
            "baseline": corpus.baseline,
            "record_count": len(corpus.rules),
        },
        records=records,
    )


def load_audit_dataset(path: str | Path) -> ConstraintAuditDataset:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ConstraintAuditDataset.from_dict(_mapping(raw, "constraint-audit root"))


def dump_audit_dataset(dataset: ConstraintAuditDataset, path: str | Path) -> None:
    dataset.validate()
    _write_yaml(Path(path), dataset.to_dict())


def freeze_intent_file(
    input_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    source = Path(input_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "mutation-intent root")
    intents = [
        MutationIntent.from_dict(_mapping(item, "mutation intent"))
        for item in root.get("intents", ())
    ]
    seen: set[str] = set()
    for intent in intents:
        if intent.intent_id in seen:
            raise ValueError(f"duplicate mutation intent id: {intent.intent_id}")
        seen.add(intent.intent_id)
    intents.sort(key=lambda item: item.intent_id)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": str(root.get("name", source.stem)),
        "metadata": dict(_mapping(root.get("metadata", {}), "metadata")),
        "intents": [intent.to_dict() for intent in intents],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    payload["frozen"] = {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "intent_count": len(intents),
    }
    _write_yaml(Path(output_path), payload)
    return payload


def build_rq1_candidate_queue(
    dataset: ConstraintAuditDataset,
    static_scan_path: str | Path,
    *,
    limit_per_constraint: int = 8,
) -> dict[str, Any]:
    if limit_per_constraint < 1:
        raise ValueError("limit_per_constraint must be positive")
    raw = json.loads(Path(static_scan_path).read_text(encoding="utf-8"))
    root = _mapping(raw, "static-scan root")
    results = _mapping(root.get("results", {}), "static-scan results")
    queue: list[dict[str, Any]] = []
    constraints_with_candidates = 0
    for record in dataset.records:
        leaves = tuple(
            parameter.rsplit(".", 1)[-1] for parameter in record.participants
        )
        target = set(leaves)
        candidates_by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for leaf in leaves:
            result = results.get(leaf)
            if not isinstance(result, Mapping):
                continue
            for raw_candidate in result.get("constraints", ()):
                candidate = _mapping(raw_candidate, "static candidate")
                review_evidence = tuple(
                    item
                    for item in candidate.get("evidence", ())
                    if _is_framework_review_evidence(
                        _mapping(item, "candidate evidence")
                    )
                )
                if not review_evidence:
                    continue
                candidate = {**dict(candidate), "evidence": review_evidence}
                parameters = tuple(
                    str(item) for item in candidate.get("parameters", ())
                )
                key = (
                    str(candidate.get("expression", "")),
                    str(candidate.get("kind", "other")),
                    tuple(sorted(parameters)),
                )
                candidates_by_key[key] = candidate
        ranked: list[dict[str, Any]] = []
        for candidate in candidates_by_key.values():
            parameters = {
                str(item).rsplit(".", 1)[-1] for item in candidate.get("parameters", ())
            }
            overlap = target & parameters
            if not overlap:
                continue
            score = _candidate_match_score(record, candidate, target, parameters)
            ranked.append(
                {
                    "score": round(score, 6),
                    "matched_participants": sorted(overlap),
                    "covers_all_participants": target <= parameters,
                    "expression": str(candidate.get("expression", "")),
                    "kind": str(candidate.get("kind", "other")),
                    "parameters": list(candidate.get("parameters", ())),
                    "confidence": float(candidate.get("confidence", 0.0)),
                    "evidence": [
                        _normalize_framework_evidence(
                            _mapping(item, "candidate evidence")
                        )
                        for item in candidate.get("evidence", ())
                    ],
                }
            )
        ranked.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["expression"]),
                tuple(item["parameters"]),
            )
        )
        selected = ranked[:limit_per_constraint]
        if selected:
            constraints_with_candidates += 1
        queue.append(
            {
                "constraint_id": record.constraint_id,
                "predicate": record.predicate,
                "guard": record.guard,
                "participants": list(record.participants),
                "review_status": "pending_manual_review",
                "candidate_count": len(ranked),
                "candidates": selected,
            }
        )
    return {
        "schema_version": 1,
        "name": f"{dataset.name}-native-validation-candidates",
        "source_audit": {
            "name": dataset.name,
            "constraint_count": len(dataset.records),
        },
        "framework_sources": dict(
            _mapping(root.get("source", {}), "static-scan source")
        ),
        "matching_policy": {
            "purpose": "review queue only; candidate scores are not coverage labels",
            "limit_per_constraint": limit_per_constraint,
            "features": [
                "participant overlap",
                "all-participant coverage",
                "predicate identifier similarity",
                "candidate confidence",
                "rejecting-guard or assertion evidence",
            ],
        },
        "constraints_with_candidates": constraints_with_candidates,
        "constraints": queue,
    }


def load_run_records(path: str | Path) -> list[ExperimentRunRecord]:
    records: list[ExperimentRunRecord] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            records.append(ExperimentRunRecord.from_dict(_mapping(raw, "run record")))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def load_historical_bugs(path: str | Path) -> list[HistoricalBugRecord]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _mapping(raw, "historical-bug root")
    records = [
        HistoricalBugRecord.from_dict(_mapping(item, "historical bug"))
        for item in root.get("bugs", ())
    ]
    seen: set[str] = set()
    for record in records:
        if record.bug_id in seen:
            raise ValueError(f"duplicate historical bug id: {record.bug_id}")
        seen.add(record.bug_id)
    return records


def summarize_rq1(
    dataset: ConstraintAuditDataset,
    run_records: Sequence[ExperimentRunRecord] = (),
    recovered_model_path: str | Path | None = None,
) -> dict[str, Any]:
    records = dataset.records
    reviewed = [
        item
        for item in records
        if item.native_validation is not ValidationCoverage.UNREVIEWED
    ]
    coverage_all = Counter(item.native_validation.value for item in records)
    coverage_reviewed = Counter(item.native_validation.value for item in reviewed)
    early = [
        item
        for item in reviewed
        if item.first_affected_milestone
        in {ExecutionMilestone.ARGUMENT_PARSING, ExecutionMilestone.CONFIG_VALIDATION}
        and item.native_validation
        in {ValidationCoverage.FULL_EXPLICIT, ValidationCoverage.PARTIAL}
    ]
    rq1_runs = [item for item in run_records if item.rq == "rq1"]
    return {
        "schema_version": 1,
        "rq": "rq1",
        "constraint_count": len(records),
        "reviewed_constraint_count": len(reviewed),
        "unreviewed_constraint_count": len(records) - len(reviewed),
        "category_counts": _counter(item.category.value for item in records),
        "semantic_class_counts": _counter(
            item.semantic_class.value for item in records
        ),
        "arity_counts": _counter(str(item.arity) for item in records),
        "guarded_count": sum(item.guard is not None for item in records),
        "guarded_rate": _ratio(
            sum(item.guard is not None for item in records), len(records)
        ),
        "software_layer_counts": _counter(
            layer.value for item in records for layer in item.software_layers
        ),
        "coverage_counts_all": dict(sorted(coverage_all.items())),
        "coverage_counts_reviewed": dict(sorted(coverage_reviewed.items())),
        "coverage_by_denominator": _coverage_by_denominator(records),
        "coverage_by_semantic_class": _coverage_by_semantic_class(records),
        "audit_complete": len(reviewed) == len(records),
        "full_coverage_rate_all": (
            _ratio(coverage_all[ValidationCoverage.FULL_EXPLICIT.value], len(records))
            if len(reviewed) == len(records)
            else None
        ),
        "full_coverage_rate_reviewed": _ratio(
            coverage_reviewed[ValidationCoverage.FULL_EXPLICIT.value], len(reviewed)
        ),
        "early_validation_rate_all": (
            _ratio(len(early), len(records)) if len(reviewed) == len(records) else None
        ),
        "early_validation_rate_reviewed": _ratio(len(early), len(reviewed)),
        "first_affected_milestone_counts": _counter(
            item.first_affected_milestone.value for item in records
        ),
        "recovered_model": (
            summarize_recovered_constraint_model(recovered_model_path)
            if recovered_model_path is not None
            else None
        ),
        "execution": _summarize_rq1_execution(dataset, rq1_runs),
    }


def summarize_recovered_constraint_model(path: str | Path) -> dict[str, Any]:
    """Summarize the lifecycle of recovered constraints after execution feedback."""

    from configfuzz.dependencies import DependencyGraph, DependencyStatus

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    root = _mapping(raw, "recovered constraint model")
    graph_payload: Mapping[str, Any]
    if isinstance(root.get("dependency_graph"), Mapping):
        graph_payload = _mapping(root["dependency_graph"], "dependency_graph")
    elif isinstance(root.get("active_validation"), Mapping) and isinstance(
        root["active_validation"].get("dependency_graph"), Mapping
    ):
        graph_payload = _mapping(
            root["active_validation"]["dependency_graph"], "dependency_graph"
        )
    else:
        graph_payload = root
    graph = DependencyGraph.from_dict(graph_payload)
    status_counts = Counter(edge.status.value for edge in graph.edges.values())
    runtime_feedback = graph.metadata.get("runtime_feedback", {})
    edge_stats = (
        runtime_feedback.get("edges", {})
        if isinstance(runtime_feedback, Mapping)
        else {}
    )
    if not isinstance(edge_stats, Mapping):
        edge_stats = {}
    valid_counterexamples = sum(
        int(stats.get("valid_counterexample", 0))
        for stats in edge_stats.values()
        if isinstance(stats, Mapping)
    )
    paired_interventions = sum(
        int(stats.get("paired_intervention", 0))
        for stats in edge_stats.values()
        if isinstance(stats, Mapping)
    )
    supported_statuses = {
        DependencyStatus.DYNAMICALLY_SUPPORTED.value,
        DependencyStatus.CONFIRMED.value,
        DependencyStatus.ENVIRONMENT_SPECIFIC.value,
    }
    supported_count = sum(status_counts[name] for name in supported_statuses)
    return {
        "constraint_count": len(graph.edges),
        "status_counts": dict(sorted(status_counts.items())),
        "execution_supported_count": supported_count,
        "confirmed_count": status_counts[DependencyStatus.CONFIRMED.value],
        "environment_specific_count": status_counts[
            DependencyStatus.ENVIRONMENT_SPECIFIC.value
        ],
        "scope_disputed_count": status_counts[DependencyStatus.SCOPE_DISPUTED.value],
        "contradicted_count": status_counts[DependencyStatus.CONTRADICTED.value],
        "remaining_static_candidate_count": status_counts[
            DependencyStatus.STATIC_CANDIDATE.value
        ],
        "valid_counterexample_count": valid_counterexamples,
        "paired_confirmation_count": paired_interventions,
    }


def summarize_rq2(
    records: Sequence[ExperimentRunRecord],
    *,
    target_milestone: ExecutionMilestone = ExecutionMilestone.OPTIMIZER_STEP,
) -> dict[str, Any]:
    grouped: dict[ExperimentMethod, list[ExperimentRunRecord]] = defaultdict(list)
    for record in records:
        if record.rq != "rq2":
            continue
        grouped[record.method].append(record)
    methods: dict[str, Any] = {}
    ordered_milestones = [
        item
        for item in ExecutionMilestone
        if MILESTONE_ORDER[item] >= 0
        and item is not ExecutionMilestone.CONFIG_GENERATION
    ]
    for method, method_records in sorted(
        grouped.items(), key=lambda item: item[0].value
    ):
        generated = [item for item in method_records if item.generated]
        deep = [
            item
            for item in method_records
            if MILESTONE_ORDER[item.deepest_milestone]
            >= MILESTONE_ORDER[target_milestone]
        ]
        intent_preserving_deep = [
            item for item in deep if item.target_value_preserved is True
        ]
        gpu_hours = sum(item.gpu_seconds for item in method_records) / 3600.0
        modification_counts = [len(item.coordinated_parameters) for item in generated]
        modification_distances = [
            item.modification_distance
            for item in generated
            if item.modification_distance is not None
        ]
        preserved = [
            item for item in generated if item.target_value_preserved is not None
        ]
        attributable_failures = [
            item
            for item in method_records
            if item.outcome
            not in {
                ExperimentOutcome.VALID,
                ExperimentOutcome.RESOURCE_FAILURE,
                ExperimentOutcome.INFRASTRUCTURE_FAILURE,
            }
        ]
        delayed_failures = [
            item
            for item in attributable_failures
            if MILESTONE_ORDER[item.deepest_milestone]
            >= MILESTONE_ORDER[ExecutionMilestone.FORWARD]
        ]
        runtime_behavior_ids = {
            value
            for item in method_records
            for value in _runtime_behavior_ids(item)
        }
        signatures = [
            signature
            for item in method_records
            if (signature := _runtime_behavior_signature(item)) is not None
        ]
        unique_signatures = set(signatures)
        methods[method.value] = {
            "run_count": len(method_records),
            "unique_intent_count": len(
                {
                    item.intent_id
                    for item in method_records
                    if item.intent_id is not None
                }
            ),
            "generated_count": len(generated),
            "not_generated_count": len(method_records) - len(generated),
            "generation_rate": _ratio(len(generated), len(method_records)),
            "target_value_retention_rate": _ratio(
                sum(item.target_value_preserved is True for item in preserved),
                len(preserved),
            ),
            "deep_execution_count": len(deep),
            "deep_execution_rate": _ratio(len(deep), len(method_records)),
            "intent_preserving_deep_execution_count": len(intent_preserving_deep),
            "intent_preserving_deep_execution_rate": _ratio(
                len(intent_preserving_deep), len(method_records)
            ),
            "gpu_hours": gpu_hours,
            "deep_execution_yield_per_gpu_hour": (
                len(deep) / gpu_hours if gpu_hours > 0 else None
            ),
            "intent_preserving_deep_execution_yield_per_gpu_hour": (
                len(intent_preserving_deep) / gpu_hours if gpu_hours > 0 else None
            ),
            "gpu_hours_per_deep_execution": (gpu_hours / len(deep) if deep else None),
            "expected_rejection_rate": _ratio(
                sum(
                    item.outcome is ExperimentOutcome.EXPECTED_REJECTION
                    for item in method_records
                ),
                len(method_records),
            ),
            "delayed_failure_rate": _ratio(
                len(delayed_failures), len(attributable_failures)
            ),
            "timeout_rate": _ratio(
                sum(item.timed_out for item in method_records), len(method_records)
            ),
            "outcome_counts": _counter(item.outcome.value for item in method_records),
            "milestone_counts": _counter(
                item.deepest_milestone.value for item in method_records
            ),
            "stage_reach_rates": {
                milestone.value: _ratio(
                    sum(
                        MILESTONE_ORDER[item.deepest_milestone]
                        >= MILESTONE_ORDER[milestone]
                        for item in method_records
                    ),
                    len(method_records),
                )
                for milestone in ordered_milestones
            },
            "coordinated_parameter_count": _distribution(modification_counts),
            "modification_distance": _distribution(modification_distances),
            "solver_seconds": _distribution(
                [item.solver_seconds for item in method_records]
            ),
            "duration_seconds": _distribution(
                [item.duration_seconds for item in method_records]
            ),
            "peak_memory_mib": _distribution(
                [
                    item.peak_memory_mib
                    for item in method_records
                    if item.peak_memory_mib is not None
                ]
            ),
            "diversity": {
                "constraints": len(
                    {
                        value
                        for item in method_records
                        for value in item.constraints_exercised
                    }
                ),
                "boundaries": len(
                    {
                        value
                        for item in method_records
                        for value in item.boundaries_exercised
                    }
                ),
                "guard_transitions": len(
                    {
                        value
                        for item in method_records
                        for value in item.guard_transitions
                    }
                ),
                "topologies": len(
                    {value for item in method_records for value in item.topologies}
                ),
                "feature_interactions": len(
                    {
                        value
                        for item in method_records
                        for value in item.feature_interactions
                    }
                ),
                "backend_paths": len(
                    {value for item in method_records for value in item.backend_paths}
                ),
                "runtime_behavior_ids": len(runtime_behavior_ids),
                "runtime_behavior_ids_per_gpu_hour": (
                    len(runtime_behavior_ids) / gpu_hours if gpu_hours > 0 else None
                ),
                "behavior_signatures": len(unique_signatures),
                "behavior_signatures_per_gpu_hour": (
                    len(unique_signatures) / gpu_hours if gpu_hours > 0 else None
                ),
                "behavior_signature_entropy_bits": _entropy_bits(signatures),
            },
        }
    return {
        "schema_version": 1,
        "rq": "rq2",
        "target_milestone": target_milestone.value,
        "intent_pool_counts": _counter(
            item.intent_pool or "unspecified"
            for item in records
            if item.rq == "rq2"
        ),
        "methods": methods,
        "static_hard_ablation": _summarize_static_hard_ablation(
            records, target_milestone
        ),
        "paired_intent_statistics": _paired_intent_ipde_statistics(
            records, target_milestone
        ),
    }


def _summarize_static_hard_ablation(
    records: Sequence[ExperimentRunRecord],
    target_milestone: ExecutionMilestone,
) -> dict[str, Any] | None:
    def key(record: ExperimentRunRecord) -> tuple[str, str | None, int | None]:
        return record.workload_id, record.intent_id, record.seed

    normal = {
        key(item): item
        for item in records
        if item.rq == "rq2" and item.method is ExperimentMethod.CONFIGFUZZ
    }
    static_hard = {
        key(item): item
        for item in records
        if item.rq == "rq2"
        and item.method is ExperimentMethod.STATIC_HARD_CONFIGFUZZ
    }
    paired_keys = sorted(set(normal) & set(static_hard), key=repr)
    if not paired_keys:
        return None

    def ipde(record: ExperimentRunRecord) -> bool:
        return bool(
            record.target_value_preserved is True
            and MILESTONE_ORDER[record.deepest_milestone]
            >= MILESTONE_ORDER[target_milestone]
        )

    normal_success = sum(ipde(normal[item]) for item in paired_keys)
    static_success = sum(ipde(static_hard[item]) for item in paired_keys)
    false_exclusions = sum(
        ipde(normal[item]) and not static_hard[item].generated for item in paired_keys
    )
    lost_deep = sum(
        ipde(normal[item]) and not ipde(static_hard[item]) for item in paired_keys
    )
    return {
        "paired_case_count": len(paired_keys),
        "configfuzz_ipde_rate": _ratio(normal_success, len(paired_keys)),
        "static_hard_ipde_rate": _ratio(static_success, len(paired_keys)),
        "paired_ipde_rate_difference": (
            (normal_success - static_success) / len(paired_keys)
        ),
        "false_exclusion_candidate_count": false_exclusions,
        "false_exclusion_candidate_rate": _ratio(false_exclusions, len(paired_keys)),
        "lost_ipde_count": lost_deep,
        "lost_ipde_rate": _ratio(lost_deep, len(paired_keys)),
    }


def _paired_intent_ipde_statistics(
    records: Sequence[ExperimentRunRecord],
    target_milestone: ExecutionMilestone,
    *,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Perform paired RQ2 inference after clustering repeated seeds by intent."""

    def success(record: ExperimentRunRecord) -> int:
        return int(
            record.target_value_preserved is True
            and MILESTONE_ORDER[record.deepest_milestone]
            >= MILESTONE_ORDER[target_milestone]
        )

    clustered: dict[
        ExperimentMethod, dict[tuple[str, str], list[int]]
    ] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.rq != "rq2" or record.intent_id is None:
            continue
        clustered[record.method][(record.workload_id, record.intent_id)].append(
            success(record)
        )

    reference = clustered.get(ExperimentMethod.CONFIGFUZZ, {})
    raw_results: list[tuple[ExperimentMethod, dict[str, Any]]] = []
    for method in ExperimentMethod:
        if method is ExperimentMethod.CONFIGFUZZ or method not in clustered:
            continue
        paired_keys = sorted(set(reference) & set(clustered[method]))
        if not paired_keys:
            continue
        differences = [
            statistics.fmean(reference[key])
            - statistics.fmean(clustered[method][key])
            for key in paired_keys
        ]
        ci_low, ci_high = _cluster_bootstrap_mean_ci(
            differences,
            samples=bootstrap_samples,
            seed=f"rq2-ipde:{method.value}",
        )
        raw_results.append(
            (
                method,
                {
                    "paired_intent_count": len(paired_keys),
                    "configfuzz_minus_baseline_ipde_rate_difference": statistics.fmean(
                        differences
                    ),
                    "cluster_bootstrap_95ci": [ci_low, ci_high],
                    "paired_sign_test_p": _paired_sign_test_pvalue(differences),
                },
            )
        )

    adjusted = _holm_adjust(
        [(method.value, result["paired_sign_test_p"]) for method, result in raw_results]
    )
    return {
        method.value: {
            **result,
            "holm_adjusted_p": adjusted.get(method.value),
        }
        for method, result in raw_results
    }


def _cluster_bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: str
) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    estimates = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(max(1, samples))
    ]
    estimates.sort()
    low_index = max(0, int(0.025 * (len(estimates) - 1)))
    high_index = min(len(estimates) - 1, int(0.975 * (len(estimates) - 1)))
    return estimates[low_index], estimates[high_index]


def _paired_sign_test_pvalue(values: Sequence[float]) -> float | None:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    n = positive + negative
    if n == 0:
        return None
    k = min(positive, negative)
    tail = sum(math.comb(n, index) for index in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _holm_adjust(
    values: Sequence[tuple[str, float | None]],
) -> dict[str, float | None]:
    valid = sorted(
        ((name, value) for name, value in values if value is not None),
        key=lambda item: float(item[1]),
    )
    adjusted: dict[str, float | None] = {name: None for name, _ in values}
    running = 0.0
    m = len(valid)
    for rank, (name, raw) in enumerate(valid):
        candidate = min(1.0, float(raw) * (m - rank))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def summarize_rq3(
    records: Sequence[ExperimentRunRecord],
    bugs: Sequence[HistoricalBugRecord],
    *,
    split: str = "evaluation",
) -> dict[str, Any]:
    benchmark_ids = {item.bug_id for item in bugs if item.split == split}
    grouped: dict[ExperimentMethod, list[ExperimentRunRecord]] = defaultdict(list)
    for record in records:
        if record.rq == "rq3":
            grouped[record.method].append(record)
    methods: dict[str, Any] = {}
    for method, method_records in sorted(
        grouped.items(), key=lambda item: item[0].value
    ):
        ordered_records = _campaign_order(method_records)
        replayed = {
            item.bug_id
            for item in ordered_records
            if _is_successful_replay(item, benchmark_ids)
        }
        confirmed_current = {
            item.bug_id
            for item in ordered_records
            if item.bug_id is not None
            and item.bug_id not in benchmark_ids
            and item.bug_status
            in {
                BugStatus.DEVELOPER_CONFIRMED,
                BugStatus.FIXED,
                BugStatus.ACCEPTED_PENDING,
            }
        }
        potential_bug_ids = {
            item.bug_id
            for item in ordered_records
            if item.bug_id is not None
            and item.outcome is ExperimentOutcome.POTENTIAL_BUG
            and item.bug_id not in benchmark_ids
        }
        rejected_bug_ids = {
            item.bug_id
            for item in ordered_records
            if item.bug_id is not None
            and item.bug_status is BugStatus.REJECTED
            and item.bug_id not in benchmark_ids
        }
        unconfirmed_bug_ids = {
            item.bug_id
            for item in ordered_records
            if item.bug_id is not None
            and item.bug_status is BugStatus.UNCONFIRMED
            and item.bug_id not in benchmark_ids
        }
        first_costs = _first_reproducer_costs(ordered_records, benchmark_ids)
        gpu_hours = sum(item.gpu_seconds for item in ordered_records) / 3600.0
        confirmed_or_replayed = replayed | confirmed_current
        methods[method.value] = {
            "run_count": len(ordered_records),
            "historical_replayed_bug_count": len(replayed),
            "historical_replayed_bug_ids": sorted(replayed),
            "bug_replay_rate": _ratio(len(replayed), len(benchmark_ids)),
            "tests_to_first_reproducer": _distribution(
                [item["tests"] for item in first_costs.values()]
            ),
            "seconds_to_first_reproducer": _distribution(
                [item["seconds"] for item in first_costs.values()]
            ),
            "gpu_hours_to_first_reproducer": _distribution(
                [item["gpu_seconds"] / 3600.0 for item in first_costs.values()]
            ),
            "first_reproducer_cost_by_bug": first_costs,
            "current_potential_bug_count": len(potential_bug_ids),
            "current_potential_bug_ids": sorted(potential_bug_ids),
            "confirmed_current_bug_count": len(confirmed_current),
            "confirmed_current_bug_ids": sorted(confirmed_current),
            "rejected_current_bug_count": len(rejected_bug_ids),
            "rejected_current_bug_ids": sorted(rejected_bug_ids),
            "unconfirmed_current_bug_count": len(unconfirmed_bug_ids),
            "unconfirmed_current_bug_ids": sorted(unconfirmed_bug_ids),
            "false_positive_rate": _ratio(
                len(rejected_bug_ids), len(potential_bug_ids)
            ),
            "gpu_hours": gpu_hours,
            "gpu_hours_per_historical_replay": (
                gpu_hours / len(replayed) if replayed else None
            ),
            "gpu_hours_per_confirmed_or_replayed_bug": (
                gpu_hours / len(confirmed_or_replayed)
                if confirmed_or_replayed
                else None
            ),
            "outcome_counts": _counter(item.outcome.value for item in ordered_records),
            "bug_status_counts": _counter(
                item.bug_status.value
                for item in ordered_records
                if item.bug_status is not None
            ),
        }
    replayed_by_any = {
        item.bug_id
        for item in records
        if item.rq == "rq3" and _is_successful_replay(item, benchmark_ids)
    }
    return {
        "schema_version": 1,
        "rq": "rq3",
        "benchmark_split": split,
        "historical_bug_count": len(benchmark_ids),
        "historical_bug_ids": sorted(benchmark_ids),
        "replayed_by_any_method_count": len(replayed_by_any),
        "unreplayed_bug_ids": sorted(benchmark_ids - replayed_by_any),
        "methods": methods,
    }


def _coverage_by_denominator(
    records: Sequence[ConstraintAuditRecord],
) -> dict[str, Any]:
    grouped: dict[str, list[ConstraintAuditRecord]] = defaultdict(list)
    for record in records:
        review = record.metadata.get("native_validation_review", {})
        denominator = (
            str(review.get("coverage_denominator", "unclassified"))
            if isinstance(review, Mapping)
            else "unclassified"
        )
        grouped[denominator].append(record)
    return {
        denominator: _coverage_group_summary(items)
        for denominator, items in sorted(grouped.items())
    }


def _coverage_by_semantic_class(
    records: Sequence[ConstraintAuditRecord],
) -> dict[str, Any]:
    grouped: dict[str, list[ConstraintAuditRecord]] = defaultdict(list)
    for record in records:
        grouped[record.semantic_class.value].append(record)
    return {
        semantic_class: _coverage_group_summary(items)
        for semantic_class, items in sorted(grouped.items())
    }


def _coverage_group_summary(
    records: Sequence[ConstraintAuditRecord],
) -> dict[str, Any]:
    counts = Counter(item.native_validation.value for item in records)
    reviewed = [
        item
        for item in records
        if item.native_validation is not ValidationCoverage.UNREVIEWED
    ]
    early = [
        item
        for item in reviewed
        if item.first_affected_milestone
        in {ExecutionMilestone.ARGUMENT_PARSING, ExecutionMilestone.CONFIG_VALIDATION}
        and item.native_validation
        in {ValidationCoverage.FULL_EXPLICIT, ValidationCoverage.PARTIAL}
    ]
    return {
        "constraint_count": len(records),
        "reviewed_count": len(reviewed),
        "audit_complete": len(reviewed) == len(records),
        "coverage_counts": dict(sorted(counts.items())),
        "full_explicit_rate": _ratio(
            counts[ValidationCoverage.FULL_EXPLICIT.value], len(reviewed)
        ),
        "early_validation_rate": _ratio(len(early), len(reviewed)),
    }


def _summarize_rq1_execution(
    dataset: ConstraintAuditDataset,
    records: Sequence[ExperimentRunRecord],
) -> dict[str, Any]:
    satisfying = [
        item for item in records if item.pair_role is ConstraintPairRole.SATISFYING
    ]
    violating = [
        item for item in records if item.pair_role is ConstraintPairRole.VIOLATING
    ]
    excluded = [
        item
        for item in violating
        if item.outcome
        in {
            ExperimentOutcome.RESOURCE_FAILURE,
            ExperimentOutcome.INFRASTRUCTURE_FAILURE,
        }
    ]
    attributable = [item for item in violating if item not in excluded]
    failures = [
        item
        for item in attributable
        if item.outcome is not ExperimentOutcome.VALID
        or item.first_failure_milestone is not None
    ]
    known_failure_stage = [
        item for item in failures if item.first_failure_milestone is not None
    ]
    late_failures = [
        item
        for item in known_failure_stage
        if MILESTONE_ORDER[item.first_failure_milestone]
        >= MILESTONE_ORDER[ExecutionMilestone.FORWARD]
    ]
    message_records = [
        item for item in failures if item.error_message_interpretable is not None
    ]
    satisfying_ids = {
        item.constraint_id for item in satisfying if item.constraint_id is not None
    }
    violating_ids = {
        item.constraint_id for item in violating if item.constraint_id is not None
    }
    audit_by_id = {item.constraint_id: item for item in dataset.records}
    return {
        "run_count": len(records),
        "satisfying_run_count": len(satisfying),
        "violating_run_count": len(violating),
        "complete_pair_count": len(satisfying_ids & violating_ids),
        "constraints_with_any_execution": len(satisfying_ids | violating_ids),
        "satisfying_pass_rate": _ratio(
            sum(item.outcome is ExperimentOutcome.VALID for item in satisfying),
            len(satisfying),
        ),
        "attributable_violating_run_count": len(attributable),
        "excluded_resource_or_infrastructure_count": len(excluded),
        "outcome_counts": _counter(item.outcome.value for item in violating),
        "first_failure_milestone_counts": _counter(
            item.first_failure_milestone.value
            if item.first_failure_milestone is not None
            else "unrecorded"
            for item in failures
        ),
        "failure_mode_counts": _counter(
            item.failure_mode.value
            if item.failure_mode is not None
            else FailureMode.UNKNOWN.value
            for item in failures
        ),
        "failure_stage_recording_rate": _ratio(len(known_failure_stage), len(failures)),
        "late_failure_rate": _ratio(len(late_failures), len(known_failure_stage)),
        "timeout_rate": _ratio(
            sum(item.timed_out for item in attributable), len(attributable)
        ),
        "interpretable_error_message_rate": _ratio(
            sum(item.error_message_interpretable is True for item in message_records),
            len(message_records),
        ),
        "time_to_detection_seconds": _distribution(
            [item.duration_seconds for item in attributable]
        ),
        "gpu_seconds_wasted": _distribution(
            [item.gpu_seconds for item in attributable]
        ),
        "peak_memory_mib": _distribution(
            [
                item.peak_memory_mib
                for item in attributable
                if item.peak_memory_mib is not None
            ]
        ),
        "cost_by_constraint_category": _rq1_cost_groups(
            attributable,
            {
                constraint_id: record.category.value
                for constraint_id, record in audit_by_id.items()
            },
        ),
        "cost_by_native_validation": _rq1_cost_groups(
            attributable,
            {
                constraint_id: record.native_validation.value
                for constraint_id, record in audit_by_id.items()
            },
        ),
    }


def _rq1_cost_groups(
    records: Sequence[ExperimentRunRecord],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    grouped: dict[str, list[ExperimentRunRecord]] = defaultdict(list)
    for record in records:
        label = labels.get(record.constraint_id or "", "unknown")
        grouped[label].append(record)
    return {
        label: {
            "run_count": len(items),
            "time_to_detection_seconds": _distribution(
                [item.duration_seconds for item in items]
            ),
            "gpu_seconds_wasted": _distribution([item.gpu_seconds for item in items]),
            "timeout_rate": _ratio(sum(item.timed_out for item in items), len(items)),
            "late_failure_rate": _ratio(
                sum(
                    item.first_failure_milestone is not None
                    and MILESTONE_ORDER[item.first_failure_milestone]
                    >= MILESTONE_ORDER[ExecutionMilestone.FORWARD]
                    for item in items
                ),
                sum(item.first_failure_milestone is not None for item in items),
            ),
        }
        for label, items in sorted(grouped.items())
    }


def _campaign_order(
    records: Sequence[ExperimentRunRecord],
) -> list[ExperimentRunRecord]:
    if records and all(item.campaign_test_index is not None for item in records):
        return sorted(records, key=lambda item: int(item.campaign_test_index or 0))
    return list(records)


def _is_successful_replay(
    record: ExperimentRunRecord,
    benchmark_ids: set[str],
) -> bool:
    return bool(
        record.bug_id in benchmark_ids
        and record.buggy_failed is True
        and record.fixed_passed is True
        and record.root_cause_match is True
    )


def _first_reproducer_costs(
    records: Sequence[ExperimentRunRecord],
    benchmark_ids: set[str],
) -> dict[str, dict[str, float | int]]:
    first: dict[str, dict[str, float | int]] = {}
    cumulative_seconds = 0.0
    cumulative_gpu_seconds = 0.0
    for fallback_index, record in enumerate(records, 1):
        cumulative_seconds += record.duration_seconds
        cumulative_gpu_seconds += record.gpu_seconds
        if not _is_successful_replay(record, benchmark_ids):
            continue
        bug_id = str(record.bug_id)
        if bug_id in first:
            continue
        first[bug_id] = {
            "tests": record.campaign_test_index or fallback_index,
            "seconds": (
                record.campaign_elapsed_seconds
                if record.campaign_elapsed_seconds is not None
                else cumulative_seconds
            ),
            "gpu_seconds": (
                record.campaign_gpu_seconds
                if record.campaign_gpu_seconds is not None
                else cumulative_gpu_seconds
            ),
        }
    return dict(sorted(first.items()))


def environment_fingerprint(repositories: Sequence[str | Path] = ()) -> dict[str, Any]:
    repos: dict[str, Any] = {}
    for raw_path in repositories:
        path = Path(raw_path).expanduser().resolve()
        repos[str(path)] = {
            "commit": _git_output(path, "rev-parse", "HEAD"),
            "branch": _git_output(path, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git_output(path, "status", "--porcelain")),
        }
    return {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "repositories": repos,
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_corpus_and_build_audit(corpus_path: str | Path) -> ConstraintAuditDataset:
    return build_rq1_audit_dataset(load_corpus(corpus_path))


def _audit_record_from_rule(rule: ManualConstraintRule) -> ConstraintAuditRecord:
    return ConstraintAuditRecord(
        constraint_id=rule.id,
        participants=rule.parameters,
        predicate=rule.expression,
        guard=rule.scope.condition,
        scope=rule.scope.to_dict(),
        arity=len(rule.parameters),
        category=_infer_category(rule),
        semantic_class=_infer_semantic_class(rule),
        provenance=tuple(source.to_dict() for source in rule.sources),
        software_layers=_infer_software_layers(rule),
        first_affected_milestone=ExecutionMilestone.UNKNOWN,
        native_validation=ValidationCoverage.UNREVIEWED,
        review_status=ReviewStatus.UNREVIEWED,
        metadata={
            "manual_enforcement": rule.enforcement.value,
            "manual_strength": rule.strength.value,
            "manual_status": rule.status.value,
            "rationale": rule.rationale,
            "repair": rule.repair,
            "classification_origin": "deterministic-bootstrap; manually review category, semantic class, framework coverage, and first affected milestone",
        },
    )


def _infer_category(rule: ManualConstraintRule) -> ConstraintCategory:
    if rule.kind.value in {"environment", "resource"} or rule.strength in {
        RuleStrength.ENVIRONMENT_LIMIT,
        RuleStrength.RESOURCE_LIMIT,
        RuleStrength.EMPIRICAL,
    }:
        return ConstraintCategory.ENVIRONMENT_DEPENDENCY
    if rule.scope.condition is not None or rule.kind.value == "conditional":
        return ConstraintCategory.FEATURE_INTERACTION
    if len(rule.parameters) == 1:
        return ConstraintCategory.LOCAL
    return ConstraintCategory.STRUCTURAL


def _infer_semantic_class(rule: ManualConstraintRule) -> SemanticClass:
    if rule.strength is RuleStrength.FRAMEWORK_REQUIREMENT:
        return SemanticClass.MATHEMATICAL_INVARIANT
    if rule.strength in {
        RuleStrength.ENVIRONMENT_LIMIT,
        RuleStrength.RESOURCE_LIMIT,
        RuleStrength.EMPIRICAL,
    }:
        return SemanticClass.RESOURCE_RISK
    if rule.strength is RuleStrength.LMSV_POLICY:
        return SemanticClass.OPTIMIZATION_PREFERENCE
    if rule.strength is RuleStrength.WORKAROUND:
        return SemanticClass.IMPLEMENTATION_LIMIT
    return SemanticClass.UNKNOWN


def _infer_software_layers(rule: ManualConstraintRule) -> tuple[SoftwareLayer, ...]:
    text = " ".join(
        [
            *(source.file for source in rule.sources),
            *(source.symbol or "" for source in rule.sources),
            *(source.source_type for source in rule.sources),
        ]
    ).lower()
    layers: list[SoftwareLayer] = []
    if any(token in text for token in ("validator", "validation", "check_")):
        layers.append(SoftwareLayer.VALIDATOR)
    if any(token in text for token in ("argparse", "argument", "cli")):
        layers.append(SoftwareLayer.ARGUMENT_PARSING)
    if any(token in text for token in ("parallel", "shard", "process_group")):
        layers.append(SoftwareLayer.PARALLEL)
    if any(
        token in text
        for token in ("communication", "collective", "all_reduce", "all_to_all")
    ):
        layers.append(SoftwareLayer.COMMUNICATION)
    if any(token in text for token in ("kernel", "aclnn", "operator", "dispatch")):
        layers.append(SoftwareLayer.KERNEL)
    if any(token in text for token in ("checkpoint", "save", "load")):
        layers.append(SoftwareLayer.CHECKPOINT)
    if any(token in text for token in ("optimizer", "train", "forward", "backward")):
        layers.append(SoftwareLayer.TRAINING)
    if any(token in text for token in ("model", "config")):
        layers.append(SoftwareLayer.MODEL_CONSTRUCTION)
    if not layers:
        layers.append(SoftwareLayer.OTHER)
    return tuple(dict.fromkeys(layers))


def _candidate_match_score(
    record: ConstraintAuditRecord,
    candidate: Mapping[str, Any],
    target_parameters: set[str],
    candidate_parameters: set[str],
) -> float:
    overlap = target_parameters & candidate_parameters
    parameter_coverage = len(overlap) / len(target_parameters)
    all_participants = 1.0 if target_parameters <= candidate_parameters else 0.0
    target_tokens = _expression_identifiers(record.predicate)
    if record.guard:
        target_tokens |= _expression_identifiers(record.guard)
    candidate_tokens = _expression_identifiers(str(candidate.get("expression", "")))
    union = target_tokens | candidate_tokens
    similarity = len(target_tokens & candidate_tokens) / len(union) if union else 0.0
    confidence = float(candidate.get("confidence", 0.0))
    evidence_strength = 0.0
    for raw_evidence in candidate.get("evidence", ()):
        evidence = _mapping(raw_evidence, "candidate evidence")
        detail = str(evidence.get("detail", "")).lower()
        if "rejecting guard" in detail or "assertion" in detail:
            evidence_strength = 1.0
            break
        if "argparse" in detail or "required" in detail:
            evidence_strength = max(evidence_strength, 0.5)
    exact = (
        1.0
        if _normalize_expression(record.predicate)
        == _normalize_expression(str(candidate.get("expression", "")))
        else 0.0
    )
    return (
        6.0 * parameter_coverage
        + 2.0 * all_participants
        + 2.0 * similarity
        + confidence
        + evidence_strength
        + 3.0 * exact
    )


def _expression_identifiers(expression: str) -> set[str]:
    ignored = {
        "and",
        "or",
        "not",
        "true",
        "false",
        "none",
        "self",
        "args",
        "model",
        "parallel",
        "training",
        "moe",
    }
    return {
        token.rsplit(".", 1)[-1].lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", expression)
        if token.rsplit(".", 1)[-1].lower() not in ignored
    }


def _normalize_expression(expression: str) -> str:
    return (
        re.sub(r"\s+", "", expression)
        .lower()
        .replace("model.", "")
        .replace("parallel.", "")
        .replace("training.", "")
        .replace("moe.", "")
    )


def _is_framework_review_evidence(evidence: Mapping[str, Any]) -> bool:
    source = str(evidence.get("source", "")).replace("\\", "/")
    return not bool(
        re.search(
            r"(^|/)(docs?|tests?(?:_extend)?|unit_tests?|system_tests?|examples?)(/|$)|README",
            source,
            re.I,
        )
    )


def _normalize_framework_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(evidence)
    source = str(data.get("source", "")).replace("\\", "/")
    for label in ("MindSpeed-LLM", "MindSpeed", "Megatron-LM"):
        marker = f"/{label}/"
        if marker in source:
            source = f"{label}/{source.split(marker, 1)[1]}"
            break
    data["source"] = source
    return data


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _runtime_behavior_ids(record: ExperimentRunRecord) -> tuple[str, ...]:
    if record.behavior_ids:
        return tuple(sorted(set(record.behavior_ids)))
    # Backward-compatible derivation for records produced before explicit behavior IDs
    # were added. Input-only constraint/boundary identifiers are intentionally excluded.
    derived = {
        *(f"topology:{value}" for value in record.topologies),
        *(f"feature:{value}" for value in record.feature_interactions),
        *(f"backend:{value}" for value in record.backend_paths),
    }
    return tuple(sorted(derived))


def _runtime_behavior_signature(record: ExperimentRunRecord) -> str | None:
    if record.behavior_signature:
        return record.behavior_signature
    behavior_ids = _runtime_behavior_ids(record)
    if not behavior_ids:
        return None
    payload = json.dumps(behavior_ids, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _entropy_bits(values: Sequence[str]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    total = len(values)
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{key} must be a non-empty string")
    return str(value).strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _string_tuple(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in values)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )


def _git_output(path: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
