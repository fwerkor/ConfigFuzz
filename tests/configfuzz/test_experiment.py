from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from configfuzz.corpus import load_corpus
from configfuzz.experiment import (
    BugStatus,
    ConstraintAuditRecord,
    ConstraintCategory,
    ConstraintPairRole,
    ExecutionMilestone,
    ExperimentMethod,
    ExperimentOutcome,
    ExperimentRunRecord,
    FailureMode,
    HistoricalBugRecord,
    ReviewStatus,
    ValidationCoverage,
    build_rq1_audit_dataset,
    build_rq1_candidate_queue,
    freeze_intent_file,
    load_audit_dataset,
    summarize_rq1,
    summarize_recovered_constraint_model,
    summarize_rq2,
    summarize_rq3,
)


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus/lmsv/manual_constraints.yaml"
PRIMARY_AUDIT = ROOT / "experiments/rq1/constraint_audit.primary.yaml"


def test_rq1_bootstrap_preserves_all_manual_rules_without_fake_coverage() -> None:
    corpus = load_corpus(CORPUS)
    dataset = build_rq1_audit_dataset(corpus)

    assert len(dataset.records) == 93
    assert {item.constraint_id for item in dataset.records} == {
        item.id for item in corpus.rules
    }
    assert all(
        item.native_validation is ValidationCoverage.UNREVIEWED
        for item in dataset.records
    )
    assert all(
        item.first_affected_milestone is ExecutionMilestone.UNKNOWN
        for item in dataset.records
    )
    assert all(
        item.review_status is ReviewStatus.UNREVIEWED for item in dataset.records
    )

    summary = summarize_rq1(dataset)
    assert summary["constraint_count"] == 93
    assert summary["reviewed_constraint_count"] == 0
    assert summary["full_coverage_rate_reviewed"] is None


def test_recovered_constraint_model_summary_tracks_validation_status(tmp_path: Path) -> None:
    from configfuzz.dependencies import (
        DependencyEdge,
        DependencyGraph,
        DependencyNode,
        DependencyNodeKind,
        DependencyRelation,
        DependencyStatus,
    )

    graph = DependencyGraph(
        nodes={
            "x": DependencyNode("x", DependencyNodeKind.PARAMETER),
            "y": DependencyNode("y", DependencyNodeKind.PARAMETER),
        },
        edges={
            "confirmed": DependencyEdge(
                id="confirmed",
                expression="x >= 1",
                predicate="x >= 1",
                relation=DependencyRelation.BOUND,
                participants=("x",),
                drivers=(),
                dependents=("x",),
                status=DependencyStatus.CONFIRMED,
            ),
            "disputed": DependencyEdge(
                id="disputed",
                expression="y >= 1",
                predicate="y >= 1",
                relation=DependencyRelation.BOUND,
                participants=("y",),
                drivers=(),
                dependents=("y",),
                status=DependencyStatus.SCOPE_DISPUTED,
            ),
        },
        metadata={
            "runtime_feedback": {
                "edges": {
                    "confirmed": {"paired_intervention": 1},
                    "disputed": {"valid_counterexample": 2},
                }
            }
        },
    )
    artifact = tmp_path / "graph.json"
    artifact.write_text(json.dumps(graph.to_dict()), encoding="utf-8")

    summary = summarize_recovered_constraint_model(artifact)
    assert summary["confirmed_count"] == 1
    assert summary["scope_disputed_count"] == 1
    assert summary["paired_confirmation_count"] == 1
    assert summary["valid_counterexample_count"] == 2


def test_rq1_summary_reports_paired_failure_costs() -> None:
    dataset = build_rq1_audit_dataset(load_corpus(CORPUS))
    constraint_id = "lmsv.task1.hidden-size-tp-divisibility"
    records = [
        _rq1_record(
            run_id="pair-pass",
            constraint_id=constraint_id,
            role=ConstraintPairRole.SATISFYING,
            outcome=ExperimentOutcome.VALID,
            duration_seconds=5.0,
            gpu_seconds=10.0,
        ),
        _rq1_record(
            run_id="pair-fail",
            constraint_id=constraint_id,
            role=ConstraintPairRole.VIOLATING,
            outcome=ExperimentOutcome.EXPECTED_REJECTION,
            duration_seconds=20.0,
            gpu_seconds=40.0,
            first_failure=ExecutionMilestone.FORWARD,
            failure_mode=FailureMode.CRASH,
            interpretable=True,
        ),
        _rq1_record(
            run_id="resource-noise",
            constraint_id="lmsv.task1.global-batch-lower-bound",
            role=ConstraintPairRole.VIOLATING,
            outcome=ExperimentOutcome.RESOURCE_FAILURE,
            duration_seconds=30.0,
            gpu_seconds=60.0,
            first_failure=ExecutionMilestone.MODEL_CONSTRUCTION,
            failure_mode=FailureMode.CRASH,
            interpretable=False,
        ),
    ]

    execution = summarize_rq1(dataset, records)["execution"]

    assert execution["complete_pair_count"] == 1
    assert execution["satisfying_pass_rate"] == 1.0
    assert execution["attributable_violating_run_count"] == 1
    assert execution["excluded_resource_or_infrastructure_count"] == 1
    assert execution["first_failure_milestone_counts"] == {"forward": 1}
    assert execution["failure_mode_counts"] == {"crash": 1}
    assert execution["late_failure_rate"] == 1.0
    assert execution["interpretable_error_message_rate"] == 1.0
    assert execution["time_to_detection_seconds"]["median"] == 20.0
    assert execution["accelerator_seconds_wasted"]["p95"] == 40.0
    assert execution["gpu_seconds_wasted"]["p95"] == 40.0
    assert execution["cost_by_constraint_category"]["structural"]["run_count"] == 1


def test_primary_audit_is_complete_and_uses_separate_denominators() -> None:
    dataset = load_audit_dataset(PRIMARY_AUDIT)
    summary = summarize_rq1(dataset)

    assert summary["reviewed_constraint_count"] == 93
    assert summary["audit_complete"] is True
    legality = summary["coverage_by_denominator"]["framework_legality"]
    assert legality["constraint_count"] == 39
    assert legality["coverage_counts"] == {
        "full_explicit": 13,
        "implicit_delayed": 8,
        "partial": 11,
        "uncovered": 7,
    }
    assert summary["coverage_by_denominator"]["policy_only"]["constraint_count"] == 52
    assert all(
        item.review_status is ReviewStatus.PRIMARY_REVIEWED for item in dataset.records
    )
    assert all(len(item.coverage_evidence) >= 1 for item in dataset.records)


def test_reviewed_coverage_requires_source_or_execution_evidence() -> None:
    corpus = load_corpus(CORPUS)
    raw = build_rq1_audit_dataset(corpus).records[0].to_dict()
    raw["native_validation"] = "full_explicit"

    with pytest.raises(ValueError, match="requires evidence"):
        ConstraintAuditRecord.from_dict(raw)


def test_freeze_intents_is_sorted_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "intents.yaml"
    first_output = tmp_path / "first.yaml"
    second_output = tmp_path / "second.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "test-intents",
                "metadata": {"campaign": "rq2"},
                "intents": [
                    {
                        "intent_id": "b",
                        "workload_id": "dense",
                        "baseline_id": "base",
                        "target_parameter": "tp",
                        "target_value": 4,
                        "intent_class": "topology",
                    },
                    {
                        "intent_id": "a",
                        "workload_id": "dense",
                        "baseline_id": "base",
                        "target_parameter": "hidden_size",
                        "target_value": 3073,
                        "intent_class": "boundary",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    first = freeze_intent_file(source, first_output)
    second = freeze_intent_file(source, second_output)

    assert [item["intent_id"] for item in first["intents"]] == ["a", "b"]
    assert first["frozen"] == second["frozen"]
    assert first_output.read_text() == second_output.read_text()


def test_rq1_candidate_queue_ranks_full_parameter_match_first(tmp_path: Path) -> None:
    dataset = build_rq1_audit_dataset(load_corpus(CORPUS))
    target = next(
        item
        for item in dataset.records
        if item.constraint_id == "lmsv.task1.hidden-size-tp-divisibility"
    )
    dataset.records = [target]
    scan = tmp_path / "scan.json"
    scan.write_text(
        json.dumps(
            {
                "source": {"MindSpeed": "abc"},
                "results": {
                    "hidden_size": {
                        "constraints": [
                            {
                                "expression": "hidden_size > 0",
                                "kind": "range",
                                "parameters": ["hidden_size"],
                                "confidence": 1.0,
                                "evidence": [],
                            },
                            {
                                "expression": "hidden_size % tensor_model_parallel_size == 0",
                                "kind": "relation",
                                "parameters": [
                                    "hidden_size",
                                    "tensor_model_parallel_size",
                                ],
                                "confidence": 0.95,
                                "evidence": [
                                    {
                                        "kind": "static",
                                        "source": "/workspace/upstream/MindSpeed/mindspeed/arguments.py",
                                        "line": 10,
                                        "detail": "assertion",
                                    }
                                ],
                            },
                        ]
                    },
                    "tensor_model_parallel_size": {"constraints": []},
                },
            }
        ),
        encoding="utf-8",
    )

    queue = build_rq1_candidate_queue(dataset, scan, limit_per_constraint=2)
    candidates = queue["constraints"][0]["candidates"]
    assert candidates[0]["covers_all_participants"] is True
    assert (
        candidates[0]["expression"] == "hidden_size % tensor_model_parallel_size == 0"
    )
    assert candidates[0]["evidence"][0]["source"].startswith("MindSpeed/")


def test_duplicate_intent_ids_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "intents.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "intents": [
                    {
                        "intent_id": "same",
                        "workload_id": "dense",
                        "baseline_id": "base",
                        "target_parameter": "tp",
                        "target_value": 2,
                        "intent_class": "topology",
                    },
                    {
                        "intent_id": "same",
                        "workload_id": "dense",
                        "baseline_id": "base",
                        "target_parameter": "tp",
                        "target_value": 4,
                        "intent_class": "topology",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate mutation intent"):
        freeze_intent_file(source, tmp_path / "frozen.yaml")


def test_rq2_summary_reports_yield_retention_and_diversity() -> None:
    records = [
        _run_record(
            run_id="cf-1",
            method=ExperimentMethod.CONFIGFUZZ,
            intent_id="i1",
            gpu_seconds=1800,
            milestone=ExecutionMilestone.OPTIMIZER_STEP,
            preserved=True,
            coordinated=("hidden_size",),
            constraints=("c1",),
        ),
        _run_record(
            run_id="cf-2",
            method=ExperimentMethod.CONFIGFUZZ,
            intent_id="i2",
            gpu_seconds=1800,
            milestone=ExecutionMilestone.BACKWARD,
            preserved=True,
            coordinated=("num_heads", "hidden_size"),
            constraints=("c2",),
        ),
        _run_record(
            run_id="raw-1",
            method=ExperimentMethod.RAW_MUTATION,
            intent_id="i1",
            gpu_seconds=3600,
            milestone=ExecutionMilestone.CONFIG_VALIDATION,
            preserved=True,
            coordinated=(),
            outcome=ExperimentOutcome.EXPECTED_REJECTION,
        ),
    ]

    summary = summarize_rq2(records)
    configfuzz = summary["methods"]["configfuzz"]
    raw = summary["methods"]["raw_mutation"]

    assert configfuzz["deep_execution_count"] == 1
    assert configfuzz["intent_preserving_deep_execution_count"] == 1
    assert configfuzz["intent_preserving_deep_execution_rate"] == 0.5
    assert configfuzz["accelerator_hours"] == 1.0
    assert configfuzz["gpu_hours"] == 1.0
    assert configfuzz["deep_execution_yield_per_accelerator_hour"] == 1.0
    assert configfuzz["deep_execution_yield_per_gpu_hour"] == 1.0
    assert configfuzz["intent_preserving_deep_execution_yield_per_accelerator_hour"] == 1.0
    assert configfuzz["intent_preserving_deep_execution_yield_per_gpu_hour"] == 1.0
    assert configfuzz["accelerator_hours_per_deep_execution"] == 1.0
    assert configfuzz["gpu_hours_per_deep_execution"] == 1.0
    assert configfuzz["target_value_retention_rate"] == 1.0
    assert configfuzz["diversity"]["constraints"] == 2
    assert configfuzz["coordinated_parameter_count"]["median"] == 1.5
    assert configfuzz["modification_distance"]["median"] == 1.0
    assert configfuzz["stage_reach_rates"]["optimizer_step"] == 0.5
    assert configfuzz["diversity"]["runtime_behavior_ids"] == 2
    assert configfuzz["diversity"]["behavior_signatures"] == 1
    assert configfuzz["diversity"]["behavior_signature_entropy_bits"] == 0.0
    assert raw["outcome_counts"] == {"expected_rejection": 1}
    assert raw["expected_rejection_rate"] == 1.0


def test_rq3_replay_requires_buggy_fixed_and_root_cause_oracle() -> None:
    bugs = [
        HistoricalBugRecord(
            bug_id="bug-1",
            split="evaluation",
            issue_or_pr="issue-1",
            buggy_commit="a" * 40,
            fixed_commit="b" * 40,
            affected_parameters=("tp", "hidden_size"),
            workload_id="dense",
            environment={},
            failure_milestone=ExecutionMilestone.FORWARD,
            failure_signature="shape mismatch",
            root_cause="incorrect shard shape",
            constraint_type=ConstraintCategory.STRUCTURAL,
            arity=2,
            oracle="fail on buggy and pass on fixed",
        )
    ]
    prelude = _run_record(
        run_id="prelude",
        method=ExperimentMethod.CONFIGFUZZ,
        intent_id="i0",
        gpu_seconds=600,
        milestone=ExecutionMilestone.BACKWARD,
        preserved=True,
        coordinated=(),
    ).to_dict()
    prelude.update({"rq": "rq3", "campaign_test_index": 1})
    replay = _run_record(
        run_id="replay",
        method=ExperimentMethod.CONFIGFUZZ,
        intent_id="i1",
        gpu_seconds=3000,
        milestone=ExecutionMilestone.FORWARD,
        preserved=True,
        coordinated=("hidden_size",),
        outcome=ExperimentOutcome.POTENTIAL_BUG,
    ).to_dict()
    replay.update(
        {
            "rq": "rq3",
            "bug_id": "bug-1",
            "buggy_failed": True,
            "fixed_passed": True,
            "root_cause_match": True,
            "bug_status": BugStatus.HISTORICAL_REPLAY.value,
            "campaign_test_index": 2,
            "campaign_elapsed_seconds": 30.0,
            "campaign_gpu_seconds": 3600.0,
        }
    )
    rejected = _run_record(
        run_id="rejected-current",
        method=ExperimentMethod.CONFIGFUZZ,
        intent_id="i2",
        gpu_seconds=0,
        milestone=ExecutionMilestone.FORWARD,
        preserved=True,
        coordinated=(),
        outcome=ExperimentOutcome.POTENTIAL_BUG,
    ).to_dict()
    rejected.update(
        {
            "rq": "rq3",
            "bug_id": "current-rejected",
            "bug_status": BugStatus.REJECTED.value,
            "campaign_test_index": 3,
        }
    )
    confirmed = _run_record(
        run_id="confirmed-current",
        method=ExperimentMethod.CONFIGFUZZ,
        intent_id="i3",
        gpu_seconds=0,
        milestone=ExecutionMilestone.CHECKPOINT_SAVE_LOAD,
        preserved=True,
        coordinated=(),
        outcome=ExperimentOutcome.POTENTIAL_BUG,
    ).to_dict()
    confirmed.update(
        {
            "rq": "rq3",
            "bug_id": "current-confirmed",
            "bug_status": BugStatus.DEVELOPER_CONFIRMED.value,
            "campaign_test_index": 4,
        }
    )
    records = [
        ExperimentRunRecord.from_dict(item)
        for item in (prelude, replay, rejected, confirmed)
    ]

    summary = summarize_rq3(records, bugs)
    method = summary["methods"]["configfuzz"]
    assert method["historical_replayed_bug_count"] == 1
    assert method["bug_replay_rate"] == 1.0
    assert method["accelerator_hours_per_historical_replay"] == 1.0
    assert method["gpu_hours_per_historical_replay"] == 1.0
    assert method["tests_to_first_reproducer"]["median"] == 2.0
    assert method["seconds_to_first_reproducer"]["median"] == 30.0
    assert method["accelerator_hours_to_first_reproducer"]["median"] == 1.0
    assert method["gpu_hours_to_first_reproducer"]["median"] == 1.0
    assert method["first_reproducer_cost_by_bug"]["bug-1"]["tests"] == 2
    assert method["confirmed_current_bug_ids"] == ["current-confirmed"]
    assert method["rejected_current_bug_ids"] == ["current-rejected"]
    assert method["false_positive_rate"] == 0.5
    assert summary["unreplayed_bug_ids"] == []


def test_jsonl_run_record_round_trip() -> None:
    record = _run_record(
        run_id="round-trip",
        method=ExperimentMethod.GLOBAL_REPAIR,
        intent_id="intent",
        gpu_seconds=0,
        milestone=ExecutionMilestone.MODEL_CONSTRUCTION,
        preserved=True,
        coordinated=("tp",),
    )

    payload = json.loads(json.dumps(record.to_dict()))
    assert payload["accelerator_seconds"] == payload["gpu_seconds"] == 0
    restored = ExperimentRunRecord.from_dict(payload)
    assert restored == record


def test_run_record_accepts_accelerator_time_fields() -> None:
    payload = _run_record(
        run_id="accelerator-time",
        method=ExperimentMethod.CONFIGFUZZ,
        intent_id="intent",
        gpu_seconds=0,
        milestone=ExecutionMilestone.FORWARD,
        preserved=True,
        coordinated=(),
    ).to_dict()
    payload.pop("gpu_seconds")
    payload.pop("campaign_gpu_seconds")
    payload["accelerator_seconds"] = 12.5
    payload["campaign_accelerator_seconds"] = 25.0

    record = ExperimentRunRecord.from_dict(payload)

    assert record.accelerator_seconds == 12.5
    assert record.gpu_seconds == 12.5
    assert record.campaign_accelerator_seconds == 25.0
    assert record.campaign_gpu_seconds == 25.0


def _rq1_record(
    *,
    run_id: str,
    constraint_id: str,
    role: ConstraintPairRole,
    outcome: ExperimentOutcome,
    duration_seconds: float,
    gpu_seconds: float,
    first_failure: ExecutionMilestone | None = None,
    failure_mode: FailureMode | None = None,
    interpretable: bool | None = None,
) -> ExperimentRunRecord:
    return ExperimentRunRecord(
        run_id=run_id,
        rq="rq1",
        method=ExperimentMethod.RAW_MUTATION,
        workload_id="dense",
        baseline_id="base",
        intent_id=None,
        seed=0,
        generated=True,
        target_value_preserved=None,
        coordinated_parameters=(),
        modification_distance=None,
        solver_seconds=0.0,
        deepest_milestone=first_failure or ExecutionMilestone.COMPLETED,
        outcome=outcome,
        duration_seconds=duration_seconds,
        gpu_seconds=gpu_seconds,
        peak_memory_mib=1024.0,
        timed_out=failure_mode is FailureMode.TIMEOUT,
        constraint_id=constraint_id,
        pair_role=role,
        first_failure_milestone=first_failure,
        failure_mode=failure_mode,
        error_message_interpretable=interpretable,
    )


def _run_record(
    *,
    run_id: str,
    method: ExperimentMethod,
    intent_id: str,
    gpu_seconds: float,
    milestone: ExecutionMilestone,
    preserved: bool,
    coordinated: tuple[str, ...],
    outcome: ExperimentOutcome = ExperimentOutcome.VALID,
    constraints: tuple[str, ...] = (),
) -> ExperimentRunRecord:
    return ExperimentRunRecord(
        run_id=run_id,
        rq="rq2",
        method=method,
        workload_id="dense",
        baseline_id="base",
        intent_id=intent_id,
        seed=0,
        generated=True,
        target_value_preserved=preserved,
        coordinated_parameters=coordinated,
        modification_distance=1.0 if coordinated else 0.0,
        solver_seconds=0.01,
        deepest_milestone=milestone,
        outcome=outcome,
        duration_seconds=10.0,
        gpu_seconds=gpu_seconds,
        peak_memory_mib=1024.0,
        timed_out=False,
        constraints_exercised=constraints,
        boundaries_exercised=("boundary",),
        guard_transitions=(),
        topologies=("tp=2",),
        feature_interactions=(),
        backend_paths=("pta",),
    )
