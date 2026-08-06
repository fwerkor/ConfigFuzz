from __future__ import annotations

import hashlib
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml


_READY_EXISTING = "ready_existing_harness"
_READY_BACKPORT = "ready_fixed_harness_backport"
_ALLOWED_READINESS = {
    _READY_EXISTING,
    _READY_BACKPORT,
    "needs_minimal_harness",
    "needs_trigger_split",
}


def build_replay_specs(
    execution_queue_path: str | Path,
    source_plan_path: str | Path,
    *,
    repository_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    queue_file = Path(execution_queue_path).expanduser().resolve()
    plan_file = Path(source_plan_path).expanduser().resolve()
    queue = yaml.safe_load(queue_file.read_text(encoding="utf-8"))
    plan = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
    if not isinstance(queue, Mapping) or not isinstance(plan, Mapping):
        raise ValueError("execution queue and replay source plan must be objects")
    candidates = queue.get("candidates")
    plans = plan.get("plans")
    roots = (
        repository_roots
        if repository_roots is not None
        else plan.get("repository_roots")
    )
    if not isinstance(candidates, list) or not isinstance(plans, list):
        raise ValueError("execution queue candidates and replay plans must be lists")
    if not isinstance(roots, Mapping):
        raise ValueError("repository roots must be supplied as an object")

    plan_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in plans:
        item = _mapping(raw, "replay plan")
        candidate_id = _required_string(item, "candidate_id")
        if candidate_id in plan_by_id:
            raise ValueError(f"duplicate replay plan: {candidate_id}")
        plan_by_id[candidate_id] = item

    queue_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in candidates:
        item = _mapping(raw, "execution candidate")
        candidate_id = _required_string(item, "candidate_id")
        if candidate_id in queue_by_id:
            raise ValueError(f"duplicate execution candidate: {candidate_id}")
        queue_by_id[candidate_id] = item
    if set(plan_by_id) != set(queue_by_id):
        raise ValueError("replay plans do not exactly match the execution queue")

    records: list[dict[str, Any]] = []
    readiness_counts: Counter[str] = Counter()
    repository_counts: Counter[str] = Counter()
    for candidate_id in sorted(
        queue_by_id, key=lambda value: int(queue_by_id[value]["rank"])
    ):
        candidate = queue_by_id[candidate_id]
        source = plan_by_id[candidate_id]
        repository = _required_string(candidate, "repository")
        root_value = roots.get(repository)
        if root_value is None:
            raise ValueError(f"{candidate_id}: no repository root for {repository}")
        root = Path(str(root_value)).expanduser().resolve()
        if not (root / ".git").is_dir():
            raise ValueError(f"{candidate_id}: repository root is not a Git checkout")
        buggy_commit = _required_string(candidate, "buggy_commit")
        fix_commit = _required_string(candidate, "fix_commit")
        _require_commit(root, buggy_commit, candidate_id)
        _require_commit(root, fix_commit, candidate_id)

        readiness = _required_string(source, "readiness")
        if readiness not in _ALLOWED_READINESS:
            raise ValueError(f"{candidate_id}: unsupported readiness {readiness!r}")
        readiness_counts[readiness] += 1
        repository_counts[repository] += 1

        harness_source = source.get("harness")
        harness = None
        if harness_source is not None:
            harness = _materialize_harness(
                _mapping(harness_source, "harness"),
                root=root,
                buggy_commit=buggy_commit,
                fix_commit=fix_commit,
                readiness=readiness,
                candidate_id=candidate_id,
            )
        elif readiness in {_READY_EXISTING, _READY_BACKPORT}:
            raise ValueError(f"{candidate_id}: ready entry must identify a harness")

        records.append(
            {
                "rank": int(candidate["rank"]),
                "candidate_id": candidate_id,
                "repository": repository,
                "subject": str(candidate.get("subject", "")),
                "buggy_commit": buggy_commit,
                "fix_commit": fix_commit,
                "claim_status": "replay_plan_only_not_executed",
                "readiness": readiness,
                "trigger_evidence_level": _required_string(
                    source, "trigger_evidence_level"
                ),
                "trigger": {
                    "assignments": dict(
                        _mapping(
                            source.get("trigger_assignments", {}), "trigger assignments"
                        )
                    ),
                    "relations": [
                        str(item) for item in source.get("trigger_relations", ())
                    ],
                },
                "harness": harness,
                "oracle": dict(_mapping(source.get("oracle", {}), "oracle")),
                "unresolved": [str(item) for item in source.get("unresolved", ())],
                "execution_verification": {
                    "buggy_runs": [],
                    "fixed_runs": [],
                    "buggy_reproduced_three_times": None,
                    "fixed_passed": None,
                    "root_cause_matched": None,
                    "minimum_configuration_recorded": None,
                    "benchmark_split": None,
                },
            }
        )

    return {
        "schema_version": 1,
        "name": "rq3-buggy-fixed-replay-specifications",
        "status": "planned_not_executed",
        "source_execution_queue": queue_file.name,
        "source_plan": plan_file.name,
        "source_plan_sha256": hashlib.sha256(plan_file.read_bytes()).hexdigest(),
        "policy": dict(_mapping(plan.get("policy", {}), "policy")),
        "candidate_count": len(records),
        "readiness_counts": dict(sorted(readiness_counts.items())),
        "repository_counts": dict(sorted(repository_counts.items())),
        "records": records,
    }


def validate_replay_specs(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("replay-spec root must be an object")
    records = raw.get("records")
    if not isinstance(records, list):
        raise ValueError("replay-spec records must be a list")
    seen: set[str] = set()
    readiness: Counter[str] = Counter()
    for raw_record in records:
        record = _mapping(raw_record, "replay-spec record")
        candidate_id = _required_string(record, "candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate replay-spec candidate: {candidate_id}")
        seen.add(candidate_id)
        if record.get("claim_status") != "replay_plan_only_not_executed":
            raise ValueError(f"{candidate_id}: replay plan must not claim execution")
        state = _required_string(record, "readiness")
        if state not in _ALLOWED_READINESS:
            raise ValueError(f"{candidate_id}: invalid readiness")
        readiness[state] += 1
        verification = _mapping(
            record.get("execution_verification"), "execution verification"
        )
        if verification.get("buggy_runs") or verification.get("fixed_runs"):
            raise ValueError(f"{candidate_id}: unexecuted replay spec contains runs")
        for key in (
            "buggy_reproduced_three_times",
            "fixed_passed",
            "root_cause_matched",
            "minimum_configuration_recorded",
            "benchmark_split",
        ):
            if verification.get(key) is not None:
                raise ValueError(f"{candidate_id}: {key} must remain unknown")
    declared = raw.get("readiness_counts")
    if not isinstance(declared, Mapping) or {
        key: int(value) for key, value in declared.items()
    } != dict(sorted(readiness.items())):
        raise ValueError("replay-spec readiness counts are inconsistent")
    if int(raw.get("candidate_count", -1)) != len(records):
        raise ValueError("replay-spec candidate_count is inconsistent")
    return {
        "valid": True,
        "candidate_count": len(records),
        "readiness_counts": dict(sorted(readiness.items())),
    }


def _materialize_harness(
    source: Mapping[str, Any],
    *,
    root: Path,
    buggy_commit: str,
    fix_commit: str,
    readiness: str,
    candidate_id: str,
) -> dict[str, Any]:
    path = _required_string(source, "path")
    fixed_exists = _git_path_exists(root, fix_commit, path)
    buggy_exists = _git_path_exists(root, buggy_commit, path)
    if readiness == _READY_EXISTING and not (fixed_exists and buggy_exists):
        raise ValueError(
            f"{candidate_id}: existing harness is not present on both commits"
        )
    if readiness == _READY_BACKPORT and not fixed_exists:
        raise ValueError(f"{candidate_id}: fixed-side harness does not exist")
    fixed_blob = (
        _git_output(root, "rev-parse", f"{fix_commit}:{path}") if fixed_exists else None
    )
    buggy_blob = (
        _git_output(root, "rev-parse", f"{buggy_commit}:{path}")
        if buggy_exists
        else None
    )
    payload = {
        "path": path,
        "mode": _required_string(source, "mode"),
        "command": _required_string(source, "command"),
        "exists_on_buggy_commit": buggy_exists,
        "exists_on_fixed_commit": fixed_exists,
        "buggy_blob": buggy_blob,
        "fixed_blob": fixed_blob,
        "backport_test_code_only": readiness == _READY_BACKPORT,
        "extraction_command": f"git show {fix_commit}:{path} > {path}",
    }
    for field in ("required_world_size", "required_device"):
        if source.get(field) is not None:
            payload[field] = source[field]
    return payload


def _git_path_exists(root: Path, commit: str, path: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{commit}:{path}"],
        capture_output=True,
        check=False,
        timeout=10,
    )
    return completed.returncode == 0


def _require_commit(root: Path, commit: str, candidate_id: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise ValueError(f"{candidate_id}: commit not found: {commit}")


def _git_output(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{key} must be a non-empty string")
    return str(value).strip()
