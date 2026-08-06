from __future__ import annotations

from pathlib import Path

import yaml

from configfuzz.bug_benchmark import build_triage_shortlist, validate_source_review


ROOT = Path(__file__).resolve().parents[2]


def test_committed_rq3_source_review_matches_execution_queue() -> None:
    result = validate_source_review(
        ROOT / "experiments/rq3/source_review.yaml",
        ROOT / "experiments/rq3/execution_queue.yaml",
    )

    assert result["record_count"] == 40
    assert result["counts"] == {
        "defer": 8,
        "exclude": 9,
        "retain_for_execution": 23,
    }
    assert result["execution_queue_count"] == 23


def test_source_review_rejects_premature_execution_claim(tmp_path: Path) -> None:
    review_path = ROOT / "experiments/rq3/source_review.yaml"
    payload = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    payload["records"][0]["execution_verification"][
        "buggy_commit_reproduced_three_times"
    ] = True
    invalid = tmp_path / "source_review.yaml"
    invalid.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    try:
        validate_source_review(invalid)
    except ValueError as exc:
        assert "must not claim execution verification" in str(exc)
    else:
        raise AssertionError("premature execution claim was accepted")


def test_shortlist_filters_low_signal_and_balances_repositories() -> None:
    candidates = [
        _candidate(
            "a",
            "MindSpeed",
            "fix: context parallel validation bug",
            ["context_parallel_size", "num_attention_heads"],
            ["mindspeed/arguments.py"],
            score=10,
        ),
        _candidate(
            "b",
            "MindSpeed-LLM",
            "[bugfix] pipeline config causes shape error",
            ["pipeline_model_parallel_size", "num_layers"],
            ["mindspeed_llm/training/arguments.py"],
            score=9,
        ),
        _candidate(
            "c",
            "MindSpeed",
            "docs: fix context parallel README",
            ["context_parallel_size"],
            ["docs/context_parallel.md"],
            score=20,
        ),
        _candidate(
            "d",
            "MindSpeed-LLM",
            "fix typo",
            ["hidden_size"],
            ["mindspeed_llm/config.py"],
            score=20,
        ),
    ]

    payload = build_triage_shortlist(candidates, limit=4)

    assert payload["shortlist_count"] == 2
    assert payload["repository_counts"] == {"MindSpeed": 1, "MindSpeed-LLM": 1}
    assert {item["candidate_id"] for item in payload["candidates"]} == {"a", "b"}
    assert all(
        item["status"] == "needs_source_review" for item in payload["candidates"]
    )
    assert all(
        item["review"]["buggy_commit_reproduced"] is None
        for item in payload["candidates"]
    )


def test_shortlist_caps_primary_parameter_repetition() -> None:
    candidates = [
        _candidate(
            f"cp-{index}",
            "MindSpeed" if index % 2 == 0 else "MindSpeed-LLM",
            f"bugfix: context parallel config error {index}",
            ["context_parallel_size"],
            [f"src/fix_{index}.py"],
            score=20 - index,
        )
        for index in range(6)
    ]

    payload = build_triage_shortlist(
        candidates,
        limit=6,
        max_per_primary_parameter=2,
    )

    assert payload["shortlist_count"] == 2
    assert payload["primary_parameter_counts"] == {"context_parallel_size": 2}


def _candidate(
    candidate_id: str,
    repository: str,
    subject: str,
    parameters: list[str],
    files: list[str],
    *,
    score: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "repository": repository,
        "fix_commit": candidate_id * 40,
        "buggy_commit": "p" + candidate_id * 39,
        "authored_at": "2026-01-01T00:00:00+00:00",
        "subject": subject,
        "changed_files": files,
        "affected_parameter_candidates": parameters,
        "patch_additions": 10,
        "patch_deletions": 5,
        "score": score,
    }
