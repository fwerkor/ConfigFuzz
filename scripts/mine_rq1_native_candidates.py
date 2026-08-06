#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from configfuzz.corpus import load_corpus
from configfuzz.experiment import (
    build_rq1_candidate_queue,
    load_audit_dataset,
    write_json,
)
from configfuzz.extractors import scan_source_paths_multi


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine native validation candidates for every RQ1 constraint."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="framework source tree; repeat for MindSpeed-LLM, MindSpeed, and Megatron-LM",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    sources = [_parse_source(item) for item in args.source]
    corpus = load_corpus(args.corpus)
    parameters = sorted(
        {
            parameter.rsplit(".", 1)[-1]
            for rule in corpus.rules
            for parameter in rule.parameters
        }
    )
    results = scan_source_paths_multi(
        [path for _, path in sources],
        parameters,
        strict=args.strict,
        jobs=args.jobs,
    )
    payload = {
        "schema_version": 1,
        "source": {
            label: {
                "path": label,
                "commit": _git_output(path, "rev-parse", "HEAD"),
                "branch": _git_output(path, "rev-parse", "--abbrev-ref", "HEAD"),
            }
            for label, path in sources
        },
        "mode": "strict" if args.strict else "broad",
        "parameters": parameters,
        "results": {
            name: _normalize_source_paths(result.to_dict(), sources)
            for name, result in results.items()
        },
    }
    write_json(args.output, payload)
    queue = build_rq1_candidate_queue(
        load_audit_dataset(args.audit),
        args.output,
        limit_per_constraint=args.limit,
    )
    write_json(args.queue_output, queue)
    print(
        json.dumps(
            {
                "parameter_count": len(parameters),
                "candidate_count": sum(
                    len(item.constraints) for item in results.values()
                ),
                "constraints_with_candidates": queue["constraints_with_candidates"],
                "scan_output": str(args.output),
                "queue_output": str(args.queue_output),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("source must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_dir():
        raise ValueError(f"invalid source: {value!r}")
    return label, path


def _normalize_source_paths(
    value: Any,
    sources: list[tuple[str, Path]],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_source_paths(item, sources) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_source_paths(item, sources) for item in value]
    if isinstance(value, tuple):
        return [_normalize_source_paths(item, sources) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for label, root in sources:
            prefix = str(root).replace("\\", "/").rstrip("/") + "/"
            if normalized.startswith(prefix):
                return f"{label}/{normalized[len(prefix) :]}"
    return value


def _git_output(path: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
