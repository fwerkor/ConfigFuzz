from __future__ import annotations

import json
from pathlib import Path

from configfuzz.dependencies import DependencyGraph
from scripts.run_rq2_reuse_campaign import run_reuse_campaign


def _case(case_id: str, method: str, value: int, *, status: str = "ready") -> dict:
    return {
        "case_id": case_id,
        "workload_id": "w",
        "baseline_id": "b",
        "intent_id": f"intent-{case_id}",
        "intent_pool": "method_independent",
        "method": method,
        "target_parameter": "x",
        "target_value": value,
        "status": status,
        "assignments": {"x": value},
        "coordinated_parameters": [],
        "target_value_preserved": True,
        "preflight": "none",
        "violated_constraints": [],
        "unknown_constraints": [],
        "solver_seconds": 0.0,
        "metadata": {"baseline_value": 1},
    }


def test_reuse_campaign_executes_each_materialized_config_once(tmp_path: Path) -> None:
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
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_count": 3,
                "intent_count": 3,
                "method_counts": {},
                "status_counts": {},
                "cases": [
                    _case("raw", "raw_mutation", 2),
                    _case("cf", "configfuzz", 2),
                    _case("filtered", "constraint_filter_only", 3, status="filtered"),
                ],
            }
        ),
        encoding="utf-8",
    )
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo CONFIGFUZZ_MILESTONE:argument_parsing\n"
        "echo CONFIGFUZZ_MILESTONE:model_construction\n"
        "echo CONFIGFUZZ_MILESTONE:forward\n"
        "echo CONFIGFUZZ_MILESTONE:backward\n"
        "echo CONFIGFUZZ_MILESTONE:optimizer_step\n"
        "echo CONFIGFUZZ_MILESTONE:completed\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    output = tmp_path / "final.jsonl"

    summary = run_reuse_campaign(
        framework="toy",
        plan=plan,
        workloads=workloads,
        launcher=launcher,
        output=output,
        runtime_root=tmp_path / "runtime",
        assembly_root=tmp_path / "assembly",
        accelerator_kind="cpu",
        devices="",
        device_count=1,
        seed=2026,
        master_port=39000,
        timeout_seconds=10.0,
        source_results=[],
        harness_paths=[launcher],
        max_infra_retries=0,
    )

    assert summary["complete"] is True
    assert summary["physical_runtime_configurations"] == 1
    runtime_rows = (tmp_path / "runtime" / "runtime-attempt-0.jsonl").read_text().splitlines()
    assert len(runtime_rows) == 1
    final_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(final_rows) == 3
    assert sum(bool(row["generated"]) for row in final_rows) == 2
    assert final_rows[1]["metadata"]["runtime_reuse"]["source_run_id"]
    assert final_rows[0]["campaign_accelerator_seconds"] == final_rows[1][
        "campaign_accelerator_seconds"
    ]
    assert final_rows[2]["campaign_accelerator_seconds"] == final_rows[1][
        "campaign_accelerator_seconds"
    ]
    assert final_rows[0]["metadata"]["accelerator_kind"] == "cpu"
