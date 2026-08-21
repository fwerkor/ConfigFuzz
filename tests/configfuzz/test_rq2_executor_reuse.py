from __future__ import annotations

from pathlib import Path

import configfuzz.rq2_executor as rq2_executor
from configfuzz.experiment import (
    ExecutionMilestone,
    ExperimentMethod,
    ExperimentOutcome,
    ExperimentRunRecord,
)
from configfuzz.rq2_executor import Qwen2PilotRuntime, execute_qwen2_pilot_cases


def _case(case_id: str, method: str, intent_id: str, hidden_size: int) -> dict:
    return {
        "case_id": case_id,
        "workload_id": "qwen2-text-train",
        "baseline_id": "qwen2-26.1",
        "intent_id": intent_id,
        "method": method,
        "status": "ready",
        "target_parameter": "hidden_size",
        "target_value": hidden_size,
        "target_value_preserved": True,
        "coordinated_parameters": [],
        "solver_seconds": 0.01,
        "violated_constraints": [],
        "assignments": {"hidden_size": hidden_size},
        "metadata": {"baseline_value": 512},
    }


def _fake_record(case: dict, index: int, cumulative: float, outcome: ExperimentOutcome) -> ExperimentRunRecord:
    return ExperimentRunRecord(
        run_id=f"rq2-pilot-{case['case_id']}",
        rq="rq2",
        method=ExperimentMethod(case["method"]),
        workload_id=case["workload_id"],
        baseline_id=case["baseline_id"],
        intent_id=case["intent_id"],
        seed=42,
        generated=True,
        target_value_preserved=True,
        coordinated_parameters=(),
        modification_distance=1.0,
        solver_seconds=0.01,
        deepest_milestone=ExecutionMilestone.OPTIMIZER_STEP,
        outcome=outcome,
        duration_seconds=1.5,
        gpu_seconds=1.5,
        peak_memory_mib=None,
        timed_out=False,
        campaign_test_index=index,
        campaign_elapsed_seconds=1.5 * index,
        campaign_gpu_seconds=cumulative + 1.5,
        solver_modifications=dict(case["assignments"]),
        metadata={"pilot_only_not_final_metrics": True},
    )


def test_npu_pilot_reuses_equivalent_effective_cli(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_execute(case, runtime, *, campaign_test_index, campaign_started, cumulative_gpu_seconds):
        calls.append(case["case_id"])
        return _fake_record(case, campaign_test_index, cumulative_gpu_seconds, ExperimentOutcome.VALID)

    monkeypatch.setattr(rq2_executor, "execute_qwen2_pilot_case", fake_execute)
    plan = {
        "cases": [
            _case("raw", "raw_mutation", "i1", 768),
            _case("cf", "configfuzz", "i2", 768),
            _case("other", "global_repair", "i3", 1024),
        ]
    }
    runtime = Qwen2PilotRuntime(launcher=tmp_path / "launch.sh", output_root=tmp_path)

    records = execute_qwen2_pilot_cases(plan, runtime)

    assert calls == ["raw", "other"]
    assert len(records) == 3
    assert records[1].method is ExperimentMethod.CONFIGFUZZ
    assert records[1].metadata["runtime_reuse"]["source_run_id"] == "rq2-pilot-raw"
    assert records[1].solver_modifications == {"hidden_size": 768}
    assert records[1].campaign_accelerator_seconds == 1.5
    assert records[-1].campaign_accelerator_seconds == 3.0


def test_npu_pilot_does_not_reuse_infrastructure_failure(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_execute(case, runtime, *, campaign_test_index, campaign_started, cumulative_gpu_seconds):
        calls.append(case["case_id"])
        outcome = (
            ExperimentOutcome.INFRASTRUCTURE_FAILURE
            if len(calls) == 1
            else ExperimentOutcome.VALID
        )
        return _fake_record(case, campaign_test_index, cumulative_gpu_seconds, outcome)

    monkeypatch.setattr(rq2_executor, "execute_qwen2_pilot_case", fake_execute)
    plan = {
        "cases": [
            _case("first", "raw_mutation", "i1", 768),
            _case("retry-equivalent", "configfuzz", "i2", 768),
        ]
    }
    runtime = Qwen2PilotRuntime(launcher=tmp_path / "launch.sh", output_root=tmp_path)

    records = execute_qwen2_pilot_cases(plan, runtime)

    assert calls == ["first", "retry-equivalent"]
    assert records[0].outcome is ExperimentOutcome.INFRASTRUCTURE_FAILURE
    assert records[1].outcome is ExperimentOutcome.VALID
    assert "runtime_reuse" not in records[1].metadata
    assert records[-1].campaign_accelerator_seconds == 3.0
