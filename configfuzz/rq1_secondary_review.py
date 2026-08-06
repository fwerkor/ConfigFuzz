from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from configfuzz.experiment import ExecutionMilestone, ValidationCoverage


_DENOMINATORS = {"framework_legality", "environment_scope", "policy_only"}
_COVERAGE_LABELS = {
    item.value
    for item in ValidationCoverage
    if item is not ValidationCoverage.UNREVIEWED
}
_MILESTONE_LABELS = {item.value for item in ExecutionMilestone}


def build_secondary_review_packet(
    audit_path: str | Path,
    primary_adjudication_path: str | Path,
    *,
    shuffle_seed: str = "configfuzz-rq1-secondary-v1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_file = Path(audit_path).expanduser().resolve()
    primary_file = Path(primary_adjudication_path).expanduser().resolve()
    audit = yaml.safe_load(audit_file.read_text(encoding="utf-8"))
    primary = yaml.safe_load(primary_file.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping) or not isinstance(primary, Mapping):
        raise ValueError("RQ1 audit and primary adjudication must be objects")
    audit_records = _records_by_id(audit.get("records"), "audit")
    primary_records = _records_by_id(primary.get("records"), "primary adjudication")
    if set(audit_records) != set(primary_records):
        raise ValueError("audit and primary adjudication constraint IDs differ")

    source_revisions = dict(
        _mapping(primary.get("source_revisions", {}), "source revisions")
    )
    ordered_ids = sorted(
        audit_records,
        key=lambda constraint_id: hashlib.sha256(
            f"{shuffle_seed}\0{constraint_id}".encode("utf-8")
        ).hexdigest(),
    )
    packet_records: list[dict[str, Any]] = []
    template_records: list[dict[str, Any]] = []
    for index, constraint_id in enumerate(ordered_ids, start=1):
        audit_record = audit_records[constraint_id]
        primary_record = primary_records[constraint_id]
        evidence = []
        for raw_evidence in primary_record.get("coverage_evidence", ()):
            item = _mapping(raw_evidence, "coverage evidence")
            sanitized = {"file": str(item.get("file", ""))}
            if item.get("lines") is not None:
                sanitized["lines"] = list(item["lines"])
            if item.get("symbol") is not None:
                sanitized["symbol"] = str(item["symbol"])
            evidence.append(sanitized)
        review_item_id = f"RQ1-{index:03d}"
        packet_record = {
            "review_item_id": review_item_id,
            "constraint_id": constraint_id,
            "participants": list(audit_record.get("participants", ())),
            "predicate": str(audit_record.get("predicate", "")),
            "guard": audit_record.get("guard"),
            "scope": dict(_mapping(audit_record.get("scope", {}), "scope")),
            "arity": int(audit_record.get("arity", 0)),
            "provenance": [
                dict(_mapping(item, "provenance"))
                for item in audit_record.get("provenance", ())
            ],
            "candidate_evidence_locations": evidence,
        }
        packet_records.append(packet_record)
        template_records.append(
            {
                "review_item_id": review_item_id,
                "constraint_id": constraint_id,
                "secondary_review": {
                    "native_validation": None,
                    "first_affected_milestone": None,
                    "coverage_denominator": None,
                    "evidence_assessment": None,
                    "notes": None,
                },
            }
        )

    common = {
        "schema_version": 1,
        "source_audit": audit_file.name,
        "source_revisions": source_revisions,
        "review_policy": {
            "primary_labels_hidden": True,
            "primary_notes_hidden": True,
            "primary_evidence_summaries_hidden": True,
            "review_source_at_recorded_locations": True,
            "allowed_native_validation": sorted(_COVERAGE_LABELS),
            "allowed_first_affected_milestone": sorted(_MILESTONE_LABELS),
            "allowed_coverage_denominator": sorted(_DENOMINATORS),
        },
    }
    packet = {
        **common,
        "name": "rq1-secondary-review-evidence-packet",
        "status": "blind_to_primary_labels",
        "record_count": len(packet_records),
        "records": packet_records,
    }
    template = {
        **common,
        "name": "rq1-secondary-review-template",
        "reviewer": None,
        "review_status": "unreviewed",
        "record_count": len(template_records),
        "records": template_records,
    }
    return packet, template


def compare_primary_secondary_reviews(
    primary_adjudication_path: str | Path,
    secondary_review_path: str | Path,
) -> dict[str, Any]:
    primary = yaml.safe_load(
        Path(primary_adjudication_path).read_text(encoding="utf-8")
    )
    secondary = yaml.safe_load(Path(secondary_review_path).read_text(encoding="utf-8"))
    if not isinstance(primary, Mapping) or not isinstance(secondary, Mapping):
        raise ValueError("primary and secondary reviews must be objects")
    primary_records = _records_by_id(primary.get("records"), "primary adjudication")
    secondary_records = _records_by_id(secondary.get("records"), "secondary review")
    if set(primary_records) != set(secondary_records):
        raise ValueError("primary and secondary review constraint IDs differ")

    field_specs = {
        "native_validation": _COVERAGE_LABELS,
        "first_affected_milestone": _MILESTONE_LABELS,
        "coverage_denominator": _DENOMINATORS,
    }
    field_results: dict[str, Any] = {}
    disagreements: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for field, allowed in field_specs.items():
        primary_values: list[str] = []
        secondary_values: list[str] = []
        field_disagreements = 0
        for constraint_id in sorted(primary_records):
            primary_value = str(primary_records[constraint_id].get(field, ""))
            review = _mapping(
                secondary_records[constraint_id].get("secondary_review", {}),
                "secondary_review",
            )
            secondary_raw = review.get(field)
            if secondary_raw is None or not str(secondary_raw).strip():
                incomplete.append(f"{constraint_id}:{field}")
                continue
            secondary_value = str(secondary_raw)
            if secondary_value not in allowed:
                raise ValueError(
                    f"{constraint_id}: invalid secondary {field}: {secondary_value!r}"
                )
            primary_values.append(primary_value)
            secondary_values.append(secondary_value)
            if primary_value != secondary_value:
                field_disagreements += 1
                disagreements.append(
                    {
                        "constraint_id": constraint_id,
                        "field": field,
                        "primary": primary_value,
                        "secondary": secondary_value,
                    }
                )
        count = len(primary_values)
        agreement_count = sum(
            left == right for left, right in zip(primary_values, secondary_values)
        )
        field_results[field] = {
            "reviewed_count": count,
            "agreement_count": agreement_count,
            "agreement_rate": agreement_count / count if count else None,
            "cohens_kappa": _cohens_kappa(primary_values, secondary_values),
            "disagreement_count": field_disagreements,
        }

    return {
        "schema_version": 1,
        "name": "rq1-primary-secondary-review-agreement",
        "secondary_reviewer": secondary.get("reviewer"),
        "complete": not incomplete,
        "incomplete_field_count": len(incomplete),
        "incomplete_fields": incomplete,
        "field_agreement": field_results,
        "disagreements": disagreements,
        "adjudication_required_count": len(
            {item["constraint_id"] for item in disagreements}
        ),
    }


def validate_secondary_review_template(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("secondary review root must be an object")
    records = _records_by_id(raw.get("records"), "secondary review")
    completed = 0
    for constraint_id, record in records.items():
        review = _mapping(record.get("secondary_review", {}), "secondary_review")
        values = (
            review.get("native_validation"),
            review.get("first_affected_milestone"),
            review.get("coverage_denominator"),
        )
        populated = [value is not None and str(value).strip() for value in values]
        if any(populated) and not all(populated):
            raise ValueError(f"{constraint_id}: secondary review is partially filled")
        if all(populated):
            if str(values[0]) not in _COVERAGE_LABELS:
                raise ValueError(f"{constraint_id}: invalid native_validation")
            if str(values[1]) not in _MILESTONE_LABELS:
                raise ValueError(f"{constraint_id}: invalid first_affected_milestone")
            if str(values[2]) not in _DENOMINATORS:
                raise ValueError(f"{constraint_id}: invalid coverage_denominator")
            completed += 1
    return {
        "valid": True,
        "record_count": len(records),
        "completed_count": completed,
        "complete": completed == len(records),
    }


def _cohens_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right):
        raise ValueError("kappa inputs must have equal length")
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def _records_by_id(value: Any, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} records must be a list")
    records: dict[str, Mapping[str, Any]] = {}
    for raw in value:
        record = _mapping(raw, f"{label} record")
        constraint_id = str(record.get("constraint_id", "")).strip()
        if not constraint_id or constraint_id in records:
            raise ValueError(f"invalid or duplicate constraint ID: {constraint_id!r}")
        records[constraint_id] = record
    return records


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value
