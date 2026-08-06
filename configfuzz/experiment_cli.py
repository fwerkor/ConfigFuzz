from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from configfuzz.experiment import (
    ExecutionMilestone,
    build_rq1_candidate_queue,
    dump_audit_dataset,
    environment_fingerprint,
    freeze_intent_file,
    load_audit_dataset,
    load_corpus_and_build_audit,
    load_historical_bugs,
    load_run_records,
    summarize_rq1,
    summarize_rq2,
    summarize_rq3,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="configfuzz-experiment",
        description="Prepare, validate, freeze, and summarize ConfigFuzz experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_rq1 = subparsers.add_parser(
        "init-rq1-audit",
        help="bootstrap the RQ1 audit table from the manual constraint corpus",
    )
    init_rq1.add_argument("corpus", type=Path)
    init_rq1.add_argument("output", type=Path)
    init_rq1.set_defaults(handler=_run_init_rq1)

    validate_rq1 = subparsers.add_parser(
        "validate-rq1-audit",
        help="validate an edited RQ1 constraint audit dataset",
    )
    validate_rq1.add_argument("audit", type=Path)
    validate_rq1.set_defaults(handler=_run_validate_rq1)

    match_rq1 = subparsers.add_parser(
        "match-rq1-candidates",
        help="rank native framework validation candidates for each RQ1 constraint",
    )
    match_rq1.add_argument("audit", type=Path)
    match_rq1.add_argument("static_scan", type=Path)
    match_rq1.add_argument("output", type=Path)
    match_rq1.add_argument("--limit", type=int, default=8)
    match_rq1.set_defaults(handler=_run_match_rq1)

    freeze = subparsers.add_parser(
        "freeze-intents",
        help="sort, validate, and fingerprint a mutation-intent set",
    )
    freeze.add_argument("input", type=Path)
    freeze.add_argument("output", type=Path)
    freeze.set_defaults(handler=_run_freeze_intents)

    validate_runs = subparsers.add_parser(
        "validate-runs",
        help="validate JSONL records produced by an experiment campaign",
    )
    validate_runs.add_argument("records", type=Path)
    validate_runs.set_defaults(handler=_run_validate_runs)

    validate_bugs = subparsers.add_parser(
        "validate-bugs",
        help="validate the historical bug benchmark dataset",
    )
    validate_bugs.add_argument("bugs", type=Path)
    validate_bugs.set_defaults(handler=_run_validate_bugs)

    summarize = subparsers.add_parser(
        "summarize",
        help="compute deterministic RQ metrics from audited data or run records",
    )
    summarize.add_argument("rq", choices=("rq1", "rq2", "rq3"))
    summarize.add_argument("input", type=Path)
    summarize.add_argument(
        "--runs",
        type=Path,
        help="optional RQ1 JSONL satisfying/violating execution records",
    )
    summarize.add_argument("--bugs", type=Path)
    summarize.add_argument(
        "--bug-split",
        choices=("development", "evaluation"),
        default="evaluation",
    )
    summarize.add_argument(
        "--target-milestone",
        choices=tuple(item.value for item in ExecutionMilestone),
        default=ExecutionMilestone.OPTIMIZER_STEP.value,
    )
    summarize.add_argument("--output", type=Path)
    summarize.set_defaults(handler=_run_summarize)

    fingerprint = subparsers.add_parser(
        "fingerprint",
        help="record the software and repository state used by a campaign",
    )
    fingerprint.add_argument(
        "--repository",
        action="append",
        default=[],
        type=Path,
        help="Git repository to fingerprint; repeatable",
    )
    fingerprint.add_argument("--output", type=Path)
    fingerprint.set_defaults(handler=_run_fingerprint)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


def _run_init_rq1(args: argparse.Namespace) -> int:
    dataset = load_corpus_and_build_audit(args.corpus)
    dump_audit_dataset(dataset, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "constraint_count": len(dataset.records),
                "warning": (
                    "native validation and first affected milestones remain unreviewed; "
                    "bootstrap classifications must be manually audited"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_validate_rq1(args: argparse.Namespace) -> int:
    dataset = load_audit_dataset(args.audit)
    print(
        json.dumps(
            {
                "valid": True,
                "constraint_count": len(dataset.records),
                "reviewed_count": sum(
                    item.native_validation.value != "unreviewed"
                    for item in dataset.records
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_match_rq1(args: argparse.Namespace) -> int:
    payload = build_rq1_candidate_queue(
        load_audit_dataset(args.audit),
        args.static_scan,
        limit_per_constraint=args.limit,
    )
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "constraint_count": len(payload["constraints"]),
                "constraints_with_candidates": payload["constraints_with_candidates"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_freeze_intents(args: argparse.Namespace) -> int:
    payload = freeze_intent_file(args.input, args.output)
    print(json.dumps(payload["frozen"], ensure_ascii=False))
    return 0


def _run_validate_runs(args: argparse.Namespace) -> int:
    records = load_run_records(args.records)
    print(json.dumps({"valid": True, "record_count": len(records)}))
    return 0


def _run_validate_bugs(args: argparse.Namespace) -> int:
    bugs = load_historical_bugs(args.bugs)
    counts = {
        split: sum(item.split == split for item in bugs)
        for split in ("development", "evaluation")
    }
    print(json.dumps({"valid": True, "bug_count": len(bugs), "splits": counts}))
    return 0


def _run_summarize(args: argparse.Namespace) -> int:
    if args.rq == "rq1":
        payload = summarize_rq1(
            load_audit_dataset(args.input),
            load_run_records(args.runs) if args.runs is not None else (),
        )
    elif args.rq == "rq2":
        payload = summarize_rq2(
            load_run_records(args.input),
            target_milestone=ExecutionMilestone(args.target_milestone),
        )
    else:
        if args.bugs is None:
            raise ValueError("RQ3 summary requires --bugs")
        payload = summarize_rq3(
            load_run_records(args.input),
            load_historical_bugs(args.bugs),
            split=args.bug_split,
        )
    if args.output is not None:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_fingerprint(args: argparse.Namespace) -> int:
    payload = environment_fingerprint(args.repository)
    if args.output is not None:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
