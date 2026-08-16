#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from configfuzz.experiment import ExperimentMethod
from configfuzz.experiment_campaign import load_frozen_intents
from configfuzz.experiment_schedule import build_replication_schedule, summarize_schedule


METHODS = (
    ExperimentMethod.RAW_MUTATION,
    ExperimentMethod.NATIVE_VALIDATOR_GUIDED,
    ExperimentMethod.CONSTRAINT_FILTER_ONLY,
    ExperimentMethod.STATIC_HARD_CONFIGFUZZ,
    ExperimentMethod.CONFIGFUZZ,
    ExperimentMethod.GLOBAL_REPAIR,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen RQ2 replication schedule for every framework subject.")
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    intents = load_frozen_intents(args.intents)
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    bindings = matrix.get("bindings", []) if isinstance(matrix, dict) else []
    supported: dict[str, set[str]] = {}
    for binding in bindings:
        if not isinstance(binding, dict) or not bool(binding.get("formal_rq2")):
            continue
        supported.setdefault(str(binding["framework_id"]), set()).add(str(binding["workload_id"]))

    subjects = []
    total_records = 0
    for subject in sorted(supported):
        schedule = build_replication_schedule(intents, methods=METHODS, supported_workloads=supported[subject])
        summary = summarize_schedule(schedule)
        total_records += int(summary["record_count"])
        subjects.append(
            {
                "framework_id": subject,
                "supported_workloads": sorted(supported[subject]),
                "summary": summary,
                "records": [item.to_dict() for item in schedule],
            }
        )

    payload = {
        "schema_version": 1,
        "name": "rq2-frozen-replication-schedule",
        "source_intents": str(args.intents),
        "source_matrix": str(args.matrix),
        "subject_count": len(subjects),
        "total_record_count": total_records,
        "subjects": subjects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"subject_count": len(subjects), "total_record_count": total_records, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
