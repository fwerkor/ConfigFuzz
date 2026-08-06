from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from configfuzz.replay_specs import build_replay_specs, validate_replay_specs


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_build_replay_specs_verifies_fixed_only_harness(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    (repository / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "buggy")
    buggy = _git(repository, "rev-parse", "HEAD")
    (repository / "test_regression.py").write_text(
        "def test_regression():\n    assert True\n", encoding="utf-8"
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fix")
    fixed = _git(repository, "rev-parse", "HEAD")

    queue = tmp_path / "queue.yaml"
    queue.write_text(
        yaml.safe_dump(
            {
                "candidates": [
                    {
                        "rank": 1,
                        "candidate_id": "repo-fix",
                        "repository": "Repo",
                        "subject": "fix",
                        "buggy_commit": buggy,
                        "fix_commit": fixed,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "policy": {"same_harness_on_buggy_and_fixed": True},
                "repository_roots": {"Repo": str(repository)},
                "plans": [
                    {
                        "candidate_id": "repo-fix",
                        "readiness": "ready_fixed_harness_backport",
                        "trigger_evidence_level": "fixed_commit_unit_test",
                        "trigger_assignments": {"feature": True},
                        "trigger_relations": ["feature triggers bug"],
                        "harness": {
                            "path": "test_regression.py",
                            "mode": "copy_fixed_test_to_buggy_checkout",
                            "command": "pytest -q test_regression.py",
                        },
                        "oracle": {
                            "kind": "exception",
                            "patterns": ["error"],
                            "fixed_expectation": "pass",
                        },
                        "unresolved": [],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    payload = build_replay_specs(
        queue,
        source,
        repository_roots={"Repo": repository},
    )
    record = payload["records"][0]

    assert record["harness"]["exists_on_buggy_commit"] is False
    assert record["harness"]["exists_on_fixed_commit"] is True
    assert record["harness"]["backport_test_code_only"] is True
    assert record["execution_verification"]["buggy_runs"] == []

    output = tmp_path / "replay.yaml"
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    assert validate_replay_specs(output)["valid"] is True
