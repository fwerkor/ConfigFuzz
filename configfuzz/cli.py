from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from configfuzz.extractors import scan_python_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="configfuzz",
        description="Infer candidate configuration constraints from source context.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="scan Python source for parameter constraints")
    scan.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Python files or directories to scan",
    )
    scan.add_argument(
        "--parameter",
        action="append",
        required=True,
        dest="parameters",
        help="parameter name to analyze; repeat for multiple parameters",
    )
    scan.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path instead of stdout",
    )
    scan.set_defaults(handler=_run_scan)
    return parser


def _run_scan(args: argparse.Namespace) -> int:
    results = [
        scan_python_paths(args.paths, parameter=parameter).to_dict()
        for parameter in args.parameters
    ]
    payload = {
        "schema_version": 1,
        "results": results,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
