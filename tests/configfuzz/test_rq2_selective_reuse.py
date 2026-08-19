from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from configfuzz.dependencies import DependencyGraph
from configfuzz.experiment import (
    ExecutionMilestone,
    ExperimentMethod,
    ExperimentOutcome,
    ExperimentRunRecord,
)
from scripts.assemble_rq2_selective_reuse import assemble


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(case_id: str, method: str, value: int, *, status: str = "ready") -> dict:
    return {
        "case_id": case_id,
        "workload_id": "w",
        "baseline_id": "b",
        "intent_id": "intent-x",
        "intent_pool": "method_independent",
        "method": method,
        "target_parameter": "x",
        "target_value": value,
        "status": status,
        "assignments": {"x": value},
        "coordinated_parameters": [],
        "target_value_preserved": True,
        "preflight": "manual_constraints_and_solver",
        "violated_constraints": [],
        "unknown_constraints": [],
        "solver_status": "sat" if status == "ready" else "unsat",
        "solver_seconds": 0.01,
        "metadata": {
            "baseline_value": 1,
            "resolved_target_parameter": "x",
            "intent_class": "boundary",
        },
    }


def _write_workload_files(tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"x": 1}), encoding="utf-8")
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(DependencyGraph().to_dict()), encoding="utf-8")
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        "\n".join(
            [
                "workloads:",
                "- workload_id: w",
                "  baseline_id: b",
                "  baseline_config: baseline.json",
                "  dependency_graph: graph.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return workloads


def test_selective_reuse_uses_identical_runtime_and_emits_one_missing_config(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    workloads = _write_workload_files(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "intent_count": 1,
                "case_count": 5,
                "method_counts": {},
                "status_counts": {},
                "cases": [
                    _case("raw", "raw_mutation", 2),
                    _case("cf", "configfuzz", 2),
                    _case("cf-new", "configfuzz", 3),
                    _case("global-new", "global_repair", 3),
                    _case("filter", "constraint_filter_only", 4, status="filtered"),
                ],
            }
        ),
        encoding="utf-8",
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source = tmp_path / "source.jsonl"
    record = ExperimentRunRecord(
        run_id="old-raw",
        rq="rq2",
        method=ExperimentMethod.RAW_MUTATION,
        workload_id="w",
        baseline_id="b",
        intent_id="source-intent",
        intent_pool="method_independent",
        seed=2026,
        generated=True,
        target_value_preserved=True,
        coordinated_parameters=(),
        modification_distance=1.0,
        solver_seconds=0.0,
        deepest_milestone=ExecutionMilestone.COMPLETED,
        outcome=ExperimentOutcome.VALID,
        duration_seconds=2.0,
        gpu_seconds=4.0,
        peak_memory_mib=None,
        timed_out=False,
        solver_modifications={"x": 2},
        metadata={
            "framework_id": "toy",
            "launcher_sha256": _sha256(launcher),
            "runner_revision": revision,
            "config_path": "/old/config.json",
            "log_path": "/old/run.log",
        },
    ).to_dict()
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    output = tmp_path / "final.jsonl"
    missing = tmp_path / "missing.json"
    manifest = tmp_path / "manifest.json"
    summary = assemble(
        framework="toy",
        plan_path=plan,
        workload_registry=workloads,
        launcher=launcher,
        source_paths=[source],
        output=output,
        missing_plan=missing,
        manifest=manifest,
        seed=2026,
    )

    assert summary["complete"] is False
    assert summary["runtime_reused_cases"] == 2
    assert summary["assembled_cases"] == 3
    assert summary["unique_missing_runtime_configurations"] == 1
    assert not output.exists()
    missing_payload = json.loads(missing.read_text(encoding="utf-8"))
    assert len(missing_payload["cases"]) == 1
    assert missing_payload["cases"][0]["assignments"] == {"x": 3}


def test_selective_reuse_writes_complete_plan_after_fresh_source_is_added(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    workloads = _write_workload_files(tmp_path)
    cases = [
        _case("raw", "raw_mutation", 2),
        _case("cf", "configfuzz", 2),
        _case("cf-new", "configfuzz", 3),
        _case("global-new", "global_repair", 3),
    ]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "intent_count": 1,
                "case_count": len(cases),
                "method_counts": {},
                "status_counts": {},
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    def write_source(path: Path, run_id: str, method: ExperimentMethod, value: int) -> None:
        payload = ExperimentRunRecord(
            run_id=run_id,
            rq="rq2",
            method=method,
            workload_id="w",
            baseline_id="b",
            intent_id="intent-x",
            intent_pool="method_independent",
            seed=2026,
            generated=True,
            target_value_preserved=True,
            coordinated_parameters=(),
            modification_distance=1.0,
            solver_seconds=0.0,
            deepest_milestone=ExecutionMilestone.COMPLETED,
            outcome=ExperimentOutcome.VALID,
            duration_seconds=2.0,
            gpu_seconds=4.0,
            peak_memory_mib=None,
            timed_out=False,
            solver_modifications={"x": value},
            metadata={
                "framework_id": "toy",
                "launcher_sha256": _sha256(launcher),
                "runner_revision": revision,
            },
        ).to_dict()
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    old = tmp_path / "old.jsonl"
    fresh = tmp_path / "fresh.jsonl"
    write_source(old, "old", ExperimentMethod.RAW_MUTATION, 2)
    write_source(fresh, "fresh", ExperimentMethod.CONFIGFUZZ, 3)
    output = tmp_path / "final.jsonl"
    summary = assemble(
        framework="toy",
        plan_path=plan,
        workload_registry=workloads,
        launcher=launcher,
        source_paths=[old, fresh],
        output=output,
        missing_plan=tmp_path / "missing.json",
        manifest=tmp_path / "manifest.json",
        seed=2026,
    )

    assert summary["complete"] is True
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert all(row["generated"] for row in rows)
    assert [row["campaign_test_index"] for row in rows] == [1, 2, 3, 4]
    assert rows[-1]["campaign_accelerator_seconds"] == 16.0
    assert rows[1]["metadata"]["runtime_reuse"]["source_run_id"] == "old"
    assert rows[3]["metadata"]["runtime_reuse"]["source_run_id"] == "fresh"


def test_selective_reuse_rejects_infrastructure_failure_source(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    workloads = _write_workload_files(tmp_path)
    case = _case("cf", "configfuzz", 2)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "intent_count": 1,
                "case_count": 1,
                "method_counts": {},
                "status_counts": {},
                "cases": [case],
            }
        ),
        encoding="utf-8",
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source = tmp_path / "infra.jsonl"
    payload = ExperimentRunRecord(
        run_id="infra",
        rq="rq2",
        method=ExperimentMethod.CONFIGFUZZ,
        workload_id="w",
        baseline_id="b",
        intent_id="intent-x",
        intent_pool="method_independent",
        seed=2026,
        generated=True,
        target_value_preserved=True,
        coordinated_parameters=(),
        modification_distance=1.0,
        solver_seconds=0.0,
        deepest_milestone=ExecutionMilestone.UNKNOWN,
        outcome=ExperimentOutcome.INFRASTRUCTURE_FAILURE,
        duration_seconds=1.0,
        gpu_seconds=2.0,
        peak_memory_mib=None,
        timed_out=False,
        solver_modifications={"x": 2},
        metadata={
            "framework_id": "toy",
            "launcher_sha256": _sha256(launcher),
            "runner_revision": revision,
        },
    ).to_dict()
    source.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    summary = assemble(
        framework="toy",
        plan_path=plan,
        workload_registry=workloads,
        launcher=launcher,
        source_paths=[source],
        output=tmp_path / "final.jsonl",
        missing_plan=tmp_path / "missing.json",
        manifest=tmp_path / "manifest.json",
        seed=2026,
    )

    assert summary["complete"] is False
    assert summary["runtime_reused_cases"] == 0
    assert summary["unique_missing_runtime_configurations"] == 1
