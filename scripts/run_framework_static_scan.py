#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from configfuzz.dependencies import DependencyGraph
from configfuzz.extractors import scan_source_paths_multi
from configfuzz.framework_profiles import get_framework_profile, list_framework_profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a versioned ConfigFuzz static scan against an external framework checkout."
    )
    parser.add_argument(
        "framework_root",
        nargs="+",
        type=Path,
        help="one or more framework Git checkouts",
    )
    parser.add_argument(
        "--profile",
        choices=[profile.key for profile in list_framework_profiles()],
        help="built-in framework profile providing source roots and default parameters",
    )
    parser.add_argument(
        "--source-subdir",
        action="append",
        default=[],
        help="source directory relative to each checkout; repeatable",
    )
    parser.add_argument("--name", help="framework display name; defaults to the profile name")
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        dest="parameters",
        help="parameter to scan; repeat for multiple parameters; profile defaults are used when omitted",
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
        components = normalized.get("components")
        if isinstance(components, list):
            normalized["components"] = [
                _normalize_component_path(item, framework_root)
                for item in components
            ]
        return normalized
    if isinstance(value, list):
        return [_normalize_sources(item, framework_root) for item in value]
    return value


def _normalize_component_path(value: Any, framework_root: Path) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return str(path.resolve().relative_to(framework_root))
    except ValueError:
        return value


def _resolve_source_roots(
    framework_roots: list[Path],
    source_subdirs: tuple[str, ...],
) -> list[Path]:
    resolved: list[Path] = []
    for framework_root in framework_roots:
        matched = False
        for subdir in source_subdirs:
            candidate = (framework_root / subdir).resolve()
            if not candidate.exists():
                continue
            matched = True
            if candidate not in resolved:
                resolved.append(candidate)
        if not matched and source_subdirs != (".",):
            continue
        if not matched:
            candidate = framework_root.resolve()
            if candidate not in resolved:
                resolved.append(candidate)
    if not resolved:
        rendered = ", ".join(source_subdirs)
        raise FileNotFoundError(f"none of the configured source subdirectories exist: {rendered}")
    return resolved


def _framework_repository_metadata(root: Path, source_roots: list[Path]) -> dict[str, Any]:
    subdirs: list[str] = []
    for source_root in source_roots:
        try:
            subdirs.append(str(source_root.relative_to(root)))
        except ValueError:
            continue
    return {
        "repository": _git_value(root, "remote", "get-url", "origin"),
        "commit": _git_value(root, "rev-parse", "HEAD"),
        "commit_date": _git_value(root, "show", "-s", "--format=%cs", "HEAD"),
        "source_subdirs": subdirs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    framework_roots = [path.resolve() for path in args.framework_root]
    for framework_root in framework_roots:
        if not framework_root.exists():
            raise FileNotFoundError(framework_root)

    profile = get_framework_profile(args.profile) if args.profile else None
    framework_name = args.name or (profile.display_name if profile else None)
    if framework_name is None:
        raise ValueError("--name is required when --profile is not supplied")
    parameters = list(dict.fromkeys(args.parameters or (profile.parameters if profile else ())))
    if not parameters:
        raise ValueError("at least one --parameter is required when --profile is not supplied")
    source_subdirs = tuple(args.source_subdir) or (profile.source_subdirs if profile else (".",))
    source_roots = _resolve_source_roots(framework_roots, source_subdirs)

    scanned = scan_source_paths_multi(
        source_roots,
        parameters,
        strict=not args.broad,
        jobs=args.jobs,
    )
    results = [scanned[parameter].to_dict() for parameter in parameters]
    for framework_root in framework_roots:
        results = _normalize_sources(results, framework_root)

    repositories = [
        _framework_repository_metadata(
            framework_root,
            [source_root for source_root in source_roots if source_root == framework_root or framework_root in source_root.parents],
        )
        for framework_root in framework_roots
    ]
    framework = {
        "name": framework_name,
        "profile": profile.key if profile else None,
        "backend": profile.backend if profile else None,
        "accelerator": profile.accelerator if profile else None,
        "role": profile.role if profile else None,
        "repositories": repositories,
        "source_subdirs": list(source_subdirs),
        "parameters": parameters,
    }
    if len(repositories) == 1:
        framework.update(repositories[0])
        framework["source_subdir"] = (
            repositories[0]["source_subdirs"][0]
            if len(repositories[0]["source_subdirs"]) == 1
            else ",".join(repositories[0]["source_subdirs"])
        )
    graph_scope = {
        "framework": str(framework["name"]),
        "source_subdir": ",".join(source_subdirs),
    }
    if profile is not None:
        graph_scope["framework_profile"] = profile.key
        graph_scope["backend"] = profile.backend
        graph_scope["accelerator"] = profile.accelerator
    commits = [item["commit"] for item in repositories if item["commit"] is not None]
    if commits:
        graph_scope["version"] = "+".join(str(commit) for commit in commits)
    dependency_graph = DependencyGraph.from_constraint_sets(
        scanned.values(),
        scope=graph_scope,
        configuration_parameters=parameters,
    ).to_dict()
    for framework_root in framework_roots:
        dependency_graph = _normalize_sources(dependency_graph, framework_root)

    kinds = Counter(
        constraint["kind"]
        for result in results
        for constraint in result["constraints"]
    )
    return {
        "schema_version": 1,
        "experiment": "framework_static_scan",
        "framework": framework,
        "scanner": {
            "mode": "broad" if args.broad else "strict",
            "jobs": args.jobs,
        },
        "summary": {
            "parameters": len(results),
            "parameters_with_candidates": sum(bool(item["constraints"]) for item in results),
            "candidates": sum(len(item["constraints"]) for item in results),
            "kinds": dict(sorted(kinds.items())),
            "dependency_nodes": dependency_graph["summary"]["nodes"],
            "dependency_edges": dependency_graph["summary"]["edges"],
            "dependency_components": dependency_graph["summary"][
                "connected_components"
            ],
        },
        "results": results,
        "dependency_graph": dependency_graph,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = perf_counter()
    payload = run(args)
    elapsed = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"parameters={payload['summary']['parameters']} "
        f"candidates={payload['summary']['candidates']} "
        f"dependency_edges={payload['summary']['dependency_edges']} "
        f"elapsed={elapsed:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
