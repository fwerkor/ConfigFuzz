#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from configfuzz.experiment import ConstraintAuditDataset, dump_audit_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply an evidence-backed primary or secondary RQ1 adjudication file."
    )
    parser.add_argument("audit", type=Path)
    parser.add_argument("adjudication", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require one adjudication entry for every constraint in the audit",
    )
    args = parser.parse_args()

    audit_raw = yaml.safe_load(args.audit.read_text(encoding="utf-8"))
    adjudication_raw = yaml.safe_load(args.adjudication.read_text(encoding="utf-8"))
    if not isinstance(audit_raw, Mapping) or not isinstance(adjudication_raw, Mapping):
        raise ValueError("audit and adjudication roots must be objects")

    records = audit_raw.get("records")
    entries = adjudication_raw.get("records")
    if not isinstance(records, list) or not isinstance(entries, list):
        raise ValueError("audit and adjudication records must be lists")

    by_id: dict[str, Mapping[str, Any]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("adjudication entry must be an object")
        constraint_id = str(raw_entry.get("constraint_id", "")).strip()
        if not constraint_id:
            raise ValueError("adjudication entry is missing constraint_id")
        if constraint_id in by_id:
            raise ValueError(f"duplicate adjudication entry: {constraint_id}")
        by_id[constraint_id] = raw_entry

    audit_ids = {
        str(item.get("constraint_id")) for item in records if isinstance(item, Mapping)
    }
    unknown_ids = sorted(set(by_id) - audit_ids)
    if unknown_ids:
        raise ValueError(f"adjudication references unknown constraints: {unknown_ids}")
    missing_ids = sorted(audit_ids - set(by_id))
    if args.require_complete and missing_ids:
        raise ValueError(f"adjudication is incomplete: {missing_ids}")

    reviewer = str(adjudication_raw.get("reviewer", "")).strip()
    review_status = str(adjudication_raw.get("review_status", "primary_reviewed"))
    if not reviewer:
        raise ValueError("adjudication reviewer must be recorded")

    changed = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("audit record must be an object")
        constraint_id = str(record.get("constraint_id"))
        entry = by_id.get(constraint_id)
        if entry is None:
            continue
        record["native_validation"] = str(entry["native_validation"])
        record["first_affected_milestone"] = str(
            entry.get("first_affected_milestone", "unknown")
        )
        evidence = entry.get("coverage_evidence", ())
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{constraint_id}: coverage_evidence must be non-empty")
        record["coverage_evidence"] = evidence
        record["review_status"] = str(entry.get("review_status", review_status))
        record["reviewers"] = [reviewer]
        note = entry.get("notes")
        if note is not None:
            record["notes"] = str(note)
        metadata = record.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{constraint_id}: metadata must be an object")
        metadata["native_validation_review"] = {
            "source_revisions": dict(adjudication_raw.get("source_revisions", {})),
            "adjudication_file": args.adjudication.name,
            "coverage_denominator": str(
                entry.get("coverage_denominator", "framework_legality")
            ),
        }
        changed += 1

    source_corpus = audit_raw.setdefault("source_corpus", {})
    if isinstance(source_corpus, dict):
        source_corpus["native_validation_review"] = {
            "reviewer": reviewer,
            "review_status": review_status,
            "source_revisions": dict(adjudication_raw.get("source_revisions", {})),
            "adjudicated_count": changed,
            "unadjudicated_count": len(records) - changed,
        }

    dataset = ConstraintAuditDataset.from_dict(audit_raw)
    dump_audit_dataset(dataset, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "adjudicated_count": changed,
                "unadjudicated_count": len(records) - changed,
                "missing_ids": missing_ids,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
