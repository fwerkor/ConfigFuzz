#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from configfuzz.extractors import scan_source_paths_multi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a versioned ConfigFuzz static scan against an external framework checkout."
    )
    parser.add_argument("framework_root", type=Path, help="framework Git checkout")
    parser.add_argument(
        "--source-subdir",
        default=".",
        help="source directory relative to the checkout",
    )
    parser.add_argument("--name", required=True, help="framework display name")
    parser.add_argument(
        "--parameter",
        action="append",
        required=True,
        dest="parameters",
        help="parameter to scan; repeat for multiple parameters",
    )
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--broad", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _git_value(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _normalize_sources(value: Any, framework_root: Path) -> Any:
    if isinstance(value, dict):
        normalized = {
            key: _normalize_sources(item, framework_root)
            for key, item in value.items()
        }
        source = normalized.get("source")
        if isinstance(source, str):
            path = Path(source)
            if path.is_absolute():
                try:
                    normalized["source"] = str(path.resolve().relative_to(framework_root))
                except ValueError:
                    pass
        return normalized
    if isinstance(value, list):
        return [_normalize_sources(item, framework_root) for item in value]
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    framework_root = args.framework_root.resolve()
    source_root = (framework_root / args.source_subdir).resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    started = perf_counter()
    scanned = scan_source_paths_multi(
        [source_root],
        args.parameters,
        strict=not args.broad,
        jobs=args.jobs,
    )
    elapsed = perf_counter() - started
    results = [scanned[parameter].to_dict() for parameter in args.parameters]
    results = _normalize_sources(results, framework_root)

    kinds = Counter(
        constraint["kind"]
        for result in results
        for constraint in result["constraints"]
    )
    return {
        "schema_version": 1,
        "experiment": "framework_static_scan",
        "framework": {
            "name": args.name,
            "repository": _git_value(framework_root, "remote", "get-url", "origin"),
            "commit": _git_value(framework_root, "rev-parse", "HEAD"),
            "commit_date": _git_value(framework_root, "show", "-s", "--format=%cs", "HEAD"),
            "source_subdir": args.source_subdir,
        },
        "scanner": {
            "mode": "broad" if args.broad else "strict",
            "jobs": args.jobs,
            "elapsed_seconds": round(elapsed, 3),
        },
        "summary": {
            "parameters": len(results),
            "parameters_with_candidates": sum(bool(item["constraints"]) for item in results),
            "candidates": sum(len(item["constraints"]) for item in results),
            "kinds": dict(sorted(kinds.items())),
        },
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"parameters={payload['summary']['parameters']} "
        f"candidates={payload['summary']['candidates']} "
        f"elapsed={payload['scanner']['elapsed_seconds']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
