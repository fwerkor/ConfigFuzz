from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from configfuzz.extractors import scan_python_paths
from configfuzz.probing import (
    ProbeManifest,
    load_samples,
    run_manifest,
    samples_payload,
)
from configfuzz.synthesis import synthesize_constraints


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

    probe = subparsers.add_parser(
        "probe",
        help="execute a runtime probing manifest and classify outcomes",
    )
    probe.add_argument("manifest", type=Path, help="JSON probe manifest")
    probe.add_argument("--output", type=Path, help="write samples to this JSON file")
    probe.set_defaults(handler=_run_probe)

    synthesize = subparsers.add_parser(
        "synthesize",
        help="synthesize a constraint set from labeled probe samples",
    )
    synthesize.add_argument("samples", type=Path, help="JSON file produced by probe")
    synthesize.add_argument(
        "--parameter",
        help="parameter to synthesize; defaults to the manifest parameter",
    )
    synthesize.add_argument("--output", type=Path, help="write the inferred specification")
    synthesize.set_defaults(handler=_run_synthesize)

    infer = subparsers.add_parser(
        "infer",
        help="run probing and synthesize a constraint set in one command",
    )
    infer.add_argument("manifest", type=Path, help="JSON probe manifest")
    infer.add_argument("--samples-output", type=Path, help="retain labeled runtime samples")
    infer.add_argument("--output", type=Path, help="write the inferred specification")
    infer.set_defaults(handler=_run_infer)
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
    _write_json(payload, args.output)
    return 0


def _run_probe(args: argparse.Namespace) -> int:
    manifest = ProbeManifest.from_path(args.manifest)
    samples = run_manifest(manifest)
    _write_json(samples_payload(manifest, samples), args.output)
    return 0


def _run_synthesize(args: argparse.Namespace) -> int:
    payload, samples = load_samples(args.samples)
    manifest = payload.get("manifest", {})
    parameter = args.parameter or manifest.get("parameter")
    if not parameter:
        raise ValueError("parameter is required when the sample file has no manifest parameter")
    context = manifest.get("context", {}) if isinstance(manifest, dict) else {}
    result = synthesize_constraints(str(parameter), samples, context=context)
    output = {
        "schema_version": 1,
        "result": result.to_dict(),
    }
    _write_json(output, args.output)
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    manifest = ProbeManifest.from_path(args.manifest)
    samples = run_manifest(manifest)
    sample_payload = samples_payload(manifest, samples)
    if args.samples_output is not None:
        _write_json(sample_payload, args.samples_output)
    result = synthesize_constraints(
        manifest.parameter,
        samples,
        context=dict(manifest.context),
    )
    output = {
        "schema_version": 1,
        "result": result.to_dict(),
    }
    _write_json(output, args.output)
    return 0


def _write_json(payload: object, path: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
