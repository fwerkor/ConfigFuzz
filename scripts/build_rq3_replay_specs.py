#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from configfuzz.replay_specs import build_replay_specs, validate_replay_specs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build source-verified RQ3 buggy/fixed replay specifications."
    )
    parser.add_argument("--execution-queue", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repository checkout used to verify commits and harness blobs; repeatable",
    )
    args = parser.parse_args()

    roots: dict[str, Path] = {}
    for value in args.repository_root:
        label, separator, path = value.partition("=")
        if not separator or not label.strip() or not path.strip():
            parser.error("--repository-root must use LABEL=PATH")
        if label in roots:
            parser.error(f"duplicate repository root label: {label}")
        roots[label] = Path(path)

    payload = build_replay_specs(
        args.execution_queue,
        args.source_plan,
        repository_roots=roots,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    result = validate_replay_specs(args.output)
    result["output"] = str(args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
