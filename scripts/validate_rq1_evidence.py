#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.experiment import load_audit_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that every RQ1 coverage citation resolves to a real file and line range."
    )
    parser.add_argument("audit", type=Path)
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="checkout root for a pinned framework source; repeatable",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="root used for repository-local artifact evidence",
    )
    args = parser.parse_args()

    roots = _parse_roots(args.source_root)
    repository_root = args.repository_root.expanduser().resolve()
    dataset = load_audit_dataset(args.audit)
    errors: list[str] = []
    checked = 0
    for record in dataset.records:
        for evidence in record.coverage_evidence:
            checked += 1
            path = _resolve_evidence_path(evidence.file, roots, repository_root)
            if path is None:
                errors.append(
                    f"{record.constraint_id}: no source root for evidence {evidence.file}"
                )
                continue
            if not path.is_file():
                errors.append(
                    f"{record.constraint_id}: evidence file does not exist: {path}"
                )
                continue
            if evidence.lines is not None:
                line_count = len(
                    path.read_text(encoding="utf-8", errors="replace").splitlines()
                )
                start, end = evidence.lines
                if start < 1 or end < start or end > line_count:
                    errors.append(
                        f"{record.constraint_id}: invalid range {start}-{end} for "
                        f"{evidence.file} ({line_count} lines)"
                    )
    if errors:
        raise ValueError("\n".join(errors))
    print(
        json.dumps(
            {
                "valid": True,
                "constraint_count": len(dataset.records),
                "evidence_count": checked,
                "source_roots": sorted(roots),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _parse_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("source root must use LABEL=PATH")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if not label or not path.is_dir():
            raise ValueError(f"invalid source root: {value!r}")
        roots[label] = path
    return roots


def _resolve_evidence_path(
    value: str,
    roots: dict[str, Path],
    repository_root: Path,
) -> Path | None:
    normalized = value.replace("\\", "/")
    if "/" in normalized:
        label, relative = normalized.split("/", 1)
        if label in roots:
            return roots[label] / relative
    local = repository_root / normalized
    return local


if __name__ == "__main__":
    raise SystemExit(main())
