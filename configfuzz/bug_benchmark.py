from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


_EXPLICIT_FIX_RE = re.compile(
    r"\bbugfix\b|\bfix(?:ed|es|ing)?\b|修复|修正|解决|异常|错误",
    re.I,
)
_LOW_SIGNAL_RE = re.compile(r"\breadme\b|\bdocs?\b|文档|资料修正|clean\s*code", re.I)
_HIGH_VALUE_RE = re.compile(
    r"parallel|并行|context|pipeline|expert|moe|checkpoint|config|参数|配置|"
    r"shape|维度|crash|hang|assert|recompute|sequence|attention",
    re.I,
)


def validate_source_review(
    review_path: str | Path,
    execution_queue_path: str | Path | None = None,
) -> dict[str, Any]:
    review_raw = yaml.safe_load(Path(review_path).read_text(encoding="utf-8"))
    if not isinstance(review_raw, Mapping):
        raise ValueError("source-review root must be an object")
    records = review_raw.get("records")
    if not isinstance(records, list):
        raise ValueError("source-review records must be a list")
    allowed = {"retain_for_execution", "defer", "exclude"}
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    retained_ids: set[str] = set()
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("source-review record must be an object")
        candidate_id = str(raw_record.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id in seen:
            raise ValueError(
                f"invalid or duplicate source-review candidate: {candidate_id!r}"
            )
        seen.add(candidate_id)
        status = str(raw_record.get("source_review_status", ""))
        if status not in allowed:
            raise ValueError(
                f"{candidate_id}: unsupported source-review status {status!r}"
            )
        counts[status] += 1
        verification = raw_record.get("execution_verification")
        if not isinstance(verification, Mapping):
            raise ValueError(
                f"{candidate_id}: execution_verification must be an object"
            )
        if any(value is not None for value in verification.values()):
            raise ValueError(
                f"{candidate_id}: source review must not claim execution verification"
            )
        if status == "retain_for_execution":
            retained_ids.add(candidate_id)
            if raw_record.get("configuration_trigger_confirmed") is not True:
                raise ValueError(f"{candidate_id}: retained trigger must be confirmed")
            for field in (
                "trigger_summary",
                "workload_evidence",
                "failure_oracle_evidence",
            ):
                if not str(raw_record.get(field, "")).strip():
                    raise ValueError(
                        f"{candidate_id}: retained entry is missing {field}"
                    )
        elif (
            status == "exclude"
            and raw_record.get("configuration_trigger_confirmed") is True
        ):
            raise ValueError(
                f"{candidate_id}: excluded entry cannot have a confirmed trigger"
            )
    declared = review_raw.get("counts", {})
    if not isinstance(declared, Mapping) or {
        key: int(declared.get(key, -1)) for key in allowed
    } != {key: counts[key] for key in allowed}:
        raise ValueError("source-review declared counts do not match records")

    result: dict[str, Any] = {
        "valid": True,
        "record_count": len(records),
        "counts": dict(sorted(counts.items())),
        "retained_candidate_ids": sorted(retained_ids),
    }
    if execution_queue_path is not None:
        queue_raw = yaml.safe_load(
            Path(execution_queue_path).read_text(encoding="utf-8")
        )
        if not isinstance(queue_raw, Mapping):
            raise ValueError("execution-queue root must be an object")
        queue_records = queue_raw.get("candidates")
        if not isinstance(queue_records, list):
            raise ValueError("execution-queue candidates must be a list")
        queue_ids = {
            str(item.get("candidate_id", ""))
            for item in queue_records
            if isinstance(item, Mapping)
        }
        if queue_ids != retained_ids:
            raise ValueError(
                "execution queue does not match retained source-review entries"
            )
        if int(queue_raw.get("candidate_count", -1)) != len(queue_ids):
            raise ValueError("execution-queue candidate_count is inconsistent")
        result["execution_queue_count"] = len(queue_ids)
    return result


def load_fix_candidates(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("fix-candidate root must be an object")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw.get("candidates", ()):
        if not isinstance(item, Mapping):
            raise ValueError("fix candidate must be an object")
        candidate = dict(item)
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id:
            repository = str(candidate.get("repository", "repo")).lower()
            commit = str(candidate.get("fix_commit", ""))
            candidate_id = f"{repository}-{commit[:12]}"
            candidate["candidate_id"] = candidate_id
        if candidate_id in seen:
            raise ValueError(f"duplicate fix candidate id: {candidate_id}")
        seen.add(candidate_id)
        candidates.append(candidate)
    return candidates


def build_triage_shortlist(
    candidates: Sequence[Mapping[str, Any]],
    *,
    limit: int = 40,
    max_per_primary_parameter: int = 5,
    max_patch_lines: int = 2000,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if max_per_primary_parameter < 1:
        raise ValueError("max_per_primary_parameter must be positive")
    eligible = [
        _ranked_candidate(item)
        for item in candidates
        if _eligible(item, max_patch_lines=max_patch_lines)
    ]
    eligible.sort(
        key=lambda item: (
            -float(item["triage_score"]),
            int(item["patch_lines"]),
            str(item["authored_at"]),
            str(item["candidate_id"]),
        )
    )

    by_repository: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        by_repository.setdefault(str(item["repository"]), []).append(item)
    repositories = sorted(by_repository)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    primary_counts: Counter[str] = Counter()

    while len(selected) < limit:
        added = False
        for repository in repositories:
            for item in by_repository[repository]:
                candidate_id = str(item["candidate_id"])
                if candidate_id in selected_ids:
                    continue
                primary = _primary_parameter(item)
                if primary_counts[primary] >= max_per_primary_parameter:
                    continue
                selected.append(_shortlist_record(item, rank=len(selected) + 1))
                selected_ids.add(candidate_id)
                primary_counts[primary] += 1
                added = True
                break
            if len(selected) >= limit:
                break
        if not added:
            break

    return {
        "schema_version": 1,
        "name": "rq3-historical-bug-triage-shortlist",
        "status": "candidate_only",
        "selection_policy": {
            "limit": limit,
            "eligible_candidate_count": len(eligible),
            "explicit_fix_language_required": True,
            "configuration_signal_required": True,
            "docs_and_tests_only_excluded": True,
            "max_patch_lines": max_patch_lines,
            "max_per_primary_parameter": max_per_primary_parameter,
            "repository_round_robin": True,
        },
        "warning": (
            "Shortlisted commits are not benchmark bugs. Each entry must still satisfy "
            "the historical-bug inclusion criteria through source review and buggy/fixed "
            "differential execution."
        ),
        "triage_checklist": [
            "identify the concrete triggering configuration parameters",
            "construct or recover a workload without giving the exact reproducer to test methods",
            "define an observable non-performance failure oracle",
            "verify failure on the parent buggy commit at least three times",
            "verify the same generated configuration passes on the fix commit",
            "confirm the patch root cause matches the observed failure",
            "minimize the triggering parameter combination",
            "assign development or evaluation split only after verification",
        ],
        "shortlist_count": len(selected),
        "repository_counts": dict(
            sorted(Counter(item["repository"] for item in selected).items())
        ),
        "primary_parameter_counts": dict(sorted(primary_counts.items())),
        "candidates": selected,
    }


def dump_triage_shortlist(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )


def _eligible(candidate: Mapping[str, Any], *, max_patch_lines: int) -> bool:
    subject = str(candidate.get("subject", ""))
    if not _EXPLICIT_FIX_RE.search(subject) or not _HIGH_VALUE_RE.search(subject):
        return False
    files = [str(item) for item in candidate.get("changed_files", ())]
    if not files:
        return False
    implementation_files = [
        item
        for item in files
        if not re.search(
            r"(^|/)(docs?|tests?(?:_extend)?|unit_tests?|system_tests?|examples?)(/|$)|README",
            item,
            re.I,
        )
    ]
    if not implementation_files:
        return False
    patch_lines = int(candidate.get("patch_additions", 0)) + int(
        candidate.get("patch_deletions", 0)
    )
    if patch_lines <= 0 or patch_lines > max_patch_lines:
        return False
    parameters = [
        str(item) for item in candidate.get("affected_parameter_candidates", ())
    ]
    return bool(
        parameters and candidate.get("buggy_commit") and candidate.get("fix_commit")
    )


def _ranked_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(candidate)
    subject = str(item.get("subject", ""))
    files = [str(value) for value in item.get("changed_files", ())]
    parameters = [str(value) for value in item.get("affected_parameter_candidates", ())]
    patch_lines = int(item.get("patch_additions", 0)) + int(
        item.get("patch_deletions", 0)
    )
    base_score = float(item.get("score", 0.0))
    score = base_score
    score += min(len(parameters), 5) * 0.5
    score += (
        2.0 if re.search(r"config|参数|配置|validation|校验", subject, re.I) else 0.0
    )
    score += (
        1.5 if re.search(r"crash|hang|assert|异常|报错|error", subject, re.I) else 0.0
    )
    score += 1.0 if re.search(r"parallel|并行|moe|checkpoint", subject, re.I) else 0.0
    score -= 1.5 if _LOW_SIGNAL_RE.search(subject) else 0.0
    score -= min(patch_lines / 1000.0, 2.0)
    score -= min(max(len(files) - 5, 0) * 0.1, 1.0)
    item["patch_lines"] = patch_lines
    item["triage_score"] = round(score, 6)
    return item


def _primary_parameter(candidate: Mapping[str, Any]) -> str:
    parameters = [
        str(item) for item in candidate.get("affected_parameter_candidates", ())
    ]
    preferred = (
        "context_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "tensor_model_parallel_size",
        "sequence_parallel",
        "num_experts",
        "moe_router_topk",
        "recompute_method",
        "recompute_num_layers",
        "hidden_size",
        "num_attention_heads",
        "micro_batch_size",
        "seq_length",
    )
    for parameter in preferred:
        if parameter in parameters:
            return parameter
    return parameters[0] if parameters else "unknown"


def _shortlist_record(candidate: Mapping[str, Any], *, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "candidate_id": candidate["candidate_id"],
        "status": "needs_source_review",
        "repository": candidate["repository"],
        "subject": candidate["subject"],
        "merge_request_number": candidate.get("merge_request_number"),
        "merge_request_url": candidate.get("merge_request_url"),
        "buggy_commit": candidate["buggy_commit"],
        "fix_commit": candidate["fix_commit"],
        "authored_at": candidate["authored_at"],
        "affected_parameter_candidates": list(
            candidate.get("affected_parameter_candidates", ())
        ),
        "primary_parameter": _primary_parameter(candidate),
        "changed_files": list(candidate.get("changed_files", ())),
        "patch_lines": candidate["patch_lines"],
        "triage_score": candidate["triage_score"],
        "review": {
            "configuration_trigger_confirmed": None,
            "workload_identified": None,
            "failure_oracle_identified": None,
            "buggy_commit_reproduced": None,
            "fixed_commit_passed": None,
            "root_cause_matched": None,
            "minimum_configuration_recorded": None,
            "benchmark_split": None,
            "notes": None,
        },
    }
