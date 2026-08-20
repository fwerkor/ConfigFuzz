from __future__ import annotations

import json
from pathlib import Path

import yaml

from configfuzz.dependencies import DependencyGraph
from configfuzz.experiment_campaign import load_campaign_workloads
from configfuzz.model import Constraint, ConstraintKind, ConstraintSet
from configfuzz.multi_target_campaign import (
    build_candidate_pools,
    execute_multi_target_campaign,
    load_candidate_intents,
    plan_multi_target_round,
)


def _graph() -> DependencyGraph:
    sets: dict[str, ConstraintSet] = {}
    for parameter in ("x", "y", "z"):
        constraint = Constraint(
            expression=f"{parameter}: integer",
            kind=ConstraintKind.TYPE,
            parameters=(parameter,),
            confidence=1.0,
        )
        sets[parameter] = ConstraintSet(parameter=parameter)
        sets[parameter].add(constraint)
    return DependencyGraph.from_constraint_sets(sets.values())


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"x": 1, "y": 2, "z": 3}\n', encoding="utf-8")
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(_graph().to_dict()), encoding="utf-8")
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "toy",
                        "baseline_id": "toy-v1",
                        "baseline_config": str(baseline),
                        "dependency_graph": str(graph),
                        "semantic_anchors": ["target_parameter"],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text(
        yaml.safe_dump(
            {
                "intents": [
                    {
                        "intent_id": f"{parameter}-{value}",
                        "workload_id": "toy",
                        "baseline_id": "toy-v1",
                        "target_parameter": parameter,
                        "target_value": value,
                        "intent_class": "test-boundary",
                        "intent_pool": "method_independent",
                    }
                    for parameter, values in {
                        "x": (4, 5),
                        "y": (6, 7),
                        "z": (8, 9),
                    }.items()
                    for value in values
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        'test -f "$1"\n'
        "echo CONFIGFUZZ_MILESTONE:argument_parsing\n"
        "echo CONFIGFUZZ_MILESTONE:forward\n"
        "echo CONFIGFUZZ_MILESTONE:optimizer_step\n"
        "echo CONFIGFUZZ_MILESTONE:completed\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return workloads, candidates, launcher


def test_round_planning_is_seed_reproducible_and_uses_distinct_targets(
    tmp_path: Path,
) -> None:
    workloads_path, candidates_path, _ = _fixture(tmp_path)
    workloads = load_campaign_workloads(workloads_path)
    pools, graphs = build_candidate_pools(
        workloads, load_candidate_intents(candidates_path)
    )

    first = plan_multi_target_round(
        round_index=7,
        seed=2026,
        targets_per_mutation=2,
        workloads=workloads,
        candidate_pools=pools,
        graphs=graphs,
    )
    second = plan_multi_target_round(
        round_index=7,
        seed=2026,
        targets_per_mutation=2,
        workloads=workloads,
        candidate_pools=pools,
        graphs=graphs,
    )

    assert first["case_id"] == second["case_id"]
    assert first["target_assignments"] == second["target_assignments"]
    assert len(first["target_assignments"]) == 2
    assert first["status"] == "ready"


def test_continuous_campaign_records_every_round_and_resumes(tmp_path: Path) -> None:
    workloads, candidates, launcher = _fixture(tmp_path)
    output = tmp_path / "results.jsonl"
    output_root = tmp_path / "runs"

    first = execute_multi_target_campaign(
        framework_id="toy-framework",
        workload_registry_path=workloads,
        candidate_path=candidates,
        launcher=launcher,
        output_root=output_root,
        output_jsonl=output,
        rounds=4,
        targets_per_mutation=2,
        seed=17,
        gpu_devices="0",
        device_count=1,
        timeout_seconds=5,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert first["written_records"] == 4
    assert first["accelerator_launches"] == 4
    assert len(rows) == 4
    assert [row["campaign_test_index"] for row in rows] == [1, 2, 3, 4]
    assert all(row["rq"] == "rq3" for row in rows)
    assert all(row["generated"] is True for row in rows)
    assert all(len(row["metadata"]["target_assignments"]) == 2 for row in rows)

    resumed = execute_multi_target_campaign(
        framework_id="toy-framework",
        workload_registry_path=workloads,
        candidate_path=candidates,
        launcher=launcher,
        output_root=output_root,
        output_jsonl=output,
        rounds=4,
        targets_per_mutation=2,
        seed=17,
        gpu_devices="0",
        device_count=1,
        timeout_seconds=5,
    )

    assert resumed["written_records"] == 0
    assert resumed["skipped_existing"] == 4
    assert len(output.read_text().splitlines()) == 4
