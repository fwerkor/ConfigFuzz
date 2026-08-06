#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from configfuzz.corpus import load_corpus
from configfuzz.experiment import write_json


FIX_RE = re.compile(r"\bfix(?:ed|es|ing)?\b|bug|修复|修正|解决|异常|错误|问题", re.I)
RELEVANT_PATH_RE = re.compile(
    r"argument|config|parallel|transformer|moe|checkpoint|attention|optimizer|training|initialize|feature",
    re.I,
)
HIGH_SIGNAL_RE = re.compile(
    r"crash|hang|assert|shape|维度|并行|parallel|checkpoint|moe|config|参数|配置",
    re.I,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine historical configuration-fix candidates from Git history."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--repository",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Git repository to mine; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repositories = [_parse_repository(item) for item in args.repository]
    corpus = load_corpus(args.corpus)
    leaves = sorted(
        {
            parameter.rsplit(".", 1)[-1]
            for rule in corpus.rules
            for parameter in rule.parameters
        },
        key=len,
        reverse=True,
    )
    records: list[dict[str, Any]] = []
    for label, path in repositories:
        records.extend(_mine_repository(label, path, leaves))
    records.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item["authored_at"]),
            str(item["fix_commit"]),
        )
    )
    payload = {
        "schema_version": 1,
        "name": "mindspeed-historical-configuration-fix-candidates",
        "policy": {
            "status": "candidate_only",
            "selection": (
                "non-merge fix-like commits touching configuration-relevant files "
                "and audited parameter names"
            ),
            "warning": (
                "Each candidate still requires a workload, failure oracle, buggy/fixed "
                "differential execution, repeatability, and root-cause confirmation."
            ),
        },
        "repositories": {
            label: {
                "commit": _git_output(path, "rev-parse", "HEAD"),
                "branch": _git_output(path, "rev-parse", "--abbrev-ref", "HEAD"),
            }
            for label, path in repositories
        },
        "candidate_count": len(records),
        "candidates": records,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "candidate_count": len(records),
                "by_repository": {
                    label: sum(item["repository"] == label for item in records)
                    for label, _ in repositories
                },
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _mine_repository(
    label: str,
    repository: Path,
    parameter_leaves: list[str],
) -> list[dict[str, Any]]:
    raw_log = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "log",
            "--no-merges",
            "--format=%H%x1f%P%x1f%aI%x1f%s%x1e",
        ],
        text=True,
        errors="replace",
    )
    records: list[dict[str, Any]] = []
    for entry in raw_log.split("\x1e"):
        if not entry.strip():
            continue
        parts = entry.strip().split("\x1f")
        if len(parts) != 4:
            continue
        commit, parents, authored_at, subject = parts
        if not FIX_RE.search(subject):
            continue
        changed_files = _git_lines(
            repository,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
        )
        relevant_files = [
            item for item in changed_files if RELEVANT_PATH_RE.search(item)
        ]
        if not relevant_files:
            continue
        patch = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "show",
                "--format=",
                "--unified=0",
                "--no-ext-diff",
                commit,
                "--",
                *relevant_files,
            ],
            text=True,
            errors="replace",
        )
        parameters = [
            leaf
            for leaf in parameter_leaves
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(leaf)}(?![A-Za-z0-9_])",
                patch,
            )
        ]
        if not parameters:
            continue
        parent = parents.split()[0] if parents else None
        merge_request_number = _merge_request_number(subject)
        additions, deletions = _numstat(repository, commit, relevant_files)
        score = 4.0 + min(len(parameters), 5) + min(len(relevant_files), 5) * 0.5
        if HIGH_SIGNAL_RE.search(subject):
            score += 2.0
        records.append(
            {
                "candidate_id": f"{label.lower()}-{commit[:12]}",
                "status": "untriaged",
                "repository": label,
                "fix_commit": commit,
                "buggy_commit": parent,
                "authored_at": authored_at,
                "subject": subject,
                "merge_request_number": merge_request_number,
                "merge_request_url": _merge_request_url(label, merge_request_number),
                "changed_files": relevant_files,
                "affected_parameter_candidates": parameters,
                "patch_additions": additions,
                "patch_deletions": deletions,
                "score": score,
            }
        )
    return records


def _parse_repository(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("repository must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label or not (path / ".git").is_dir():
        raise ValueError(f"invalid repository: {value!r}")
    return label, path


def _merge_request_number(subject: str) -> int | None:
    match = re.search(r"!(\d+)", subject)
    return int(match.group(1)) if match else None


def _merge_request_url(label: str, number: int | None) -> str | None:
    if number is None:
        return None
    repository = {
        "MindSpeed-LLM": "MindSpeed-LLM",
        "MindSpeed": "MindSpeed",
    }.get(label)
    if repository is None:
        return None
    return f"https://gitee.com/ascend/{repository}/pulls/{number}"


def _numstat(repository: Path, commit: str, files: list[str]) -> tuple[int, int]:
    lines = _git_lines(
        repository,
        "show",
        "--format=",
        "--numstat",
        commit,
        "--",
        *files,
    )
    additions = 0
    deletions = 0
    for line in lines:
        parts = line.split("\t", 2)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        additions += int(parts[0])
        deletions += int(parts[1])
    return additions, deletions


def _git_lines(repository: Path, *args: str) -> list[str]:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args],
        text=True,
        errors="replace",
    ).splitlines()


def _git_output(repository: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
