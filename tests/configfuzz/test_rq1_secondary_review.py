from __future__ import annotations

from pathlib import Path

import yaml

from configfuzz.rq1_secondary_review import (
    build_secondary_review_packet,
    compare_primary_secondary_reviews,
    validate_secondary_review_template,
)


def test_secondary_packet_hides_primary_labels_and_compares_reviews(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.yaml"
    audit.write_text(
        yaml.safe_dump(
            {
                "records": [
                    {
                        "constraint_id": "c1",
                        "participants": ["x"],
                        "predicate": "x >= 1",
                        "guard": None,
                        "scope": {},
                        "arity": 1,
                        "provenance": [{"file": "validator.py", "lines": [1, 2]}],
                    },
                    {
                        "constraint_id": "c2",
                        "participants": ["y"],
                        "predicate": "y == true",
                        "guard": None,
                        "scope": {},
                        "arity": 1,
                        "provenance": [{"file": "validator.py", "lines": [3, 4]}],
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    primary = tmp_path / "primary.yaml"
    primary.write_text(
        yaml.safe_dump(
            {
                "source_revisions": {"Repo": "abc"},
                "records": [
                    {
                        "constraint_id": "c1",
                        "native_validation": "full_explicit",
                        "first_affected_milestone": "config_validation",
                        "coverage_denominator": "framework_legality",
                        "coverage_evidence": [
                            {
                                "file": "Repo/validator.py",
                                "lines": [10, 12],
                                "detail": "primary interpretation must be hidden",
                            }
                        ],
                        "notes": "primary notes must be hidden",
                    },
                    {
                        "constraint_id": "c2",
                        "native_validation": "uncovered",
                        "first_affected_milestone": "unknown",
                        "coverage_denominator": "policy_only",
                        "coverage_evidence": [],
                        "notes": "hidden",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    packet, template = build_secondary_review_packet(audit, primary)

    assert packet["record_count"] == 2
    assert "native_validation" not in packet["records"][0]
    assert "detail" not in packet["records"][0]["candidate_evidence_locations"][0]
    assert template["records"][0]["secondary_review"]["native_validation"] is None

    by_id = {item["constraint_id"]: item for item in template["records"]}
    by_id["c1"]["secondary_review"].update(
        {
            "native_validation": "full_explicit",
            "first_affected_milestone": "config_validation",
            "coverage_denominator": "framework_legality",
        }
    )
    by_id["c2"]["secondary_review"].update(
        {
            "native_validation": "partial",
            "first_affected_milestone": "model_construction",
            "coverage_denominator": "policy_only",
        }
    )
    template["reviewer"] = "secondary"
    secondary = tmp_path / "secondary.yaml"
    secondary.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")

    assert validate_secondary_review_template(secondary)["complete"] is True
    comparison = compare_primary_secondary_reviews(primary, secondary)
    assert comparison["complete"] is True
    assert comparison["adjudication_required_count"] == 1
    assert (
        comparison["field_agreement"]["coverage_denominator"]["agreement_rate"] == 1.0
    )
