from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from configfuzz.dependencies import (
    DependencyEdge,
    DependencyGraph,
    DependencyNode,
    DependencyNodeKind,
    DependencyRelation,
    DependencyStatus,
)
from configfuzz.experiment import ExperimentMethod, MutationIntent, freeze_intent_file
from configfuzz.experiment_campaign import (
    load_campaign_workloads,
    load_frozen_intents,
    plan_campaign,
)


def test_plan_campaign_expands_all_methods_and_global_ablation(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"x": 8, "y": 2, "z": 0}), encoding="utf-8")
    graph = DependencyGraph(
        nodes={
            name: DependencyNode(name=name, kind=DependencyNodeKind.PARAMETER)
            for name in ("x", "y", "z")
        },
        edges={
            "xy": DependencyEdge(
                id="xy",
                expression="x % y == 0",
                predicate="x % y == 0",
                relation=DependencyRelation.DIVISIBILITY,
                participants=("x", "y"),
                drivers=("x",),
                dependents=("y",),
                status=DependencyStatus.CONFIRMED,
            ),
            "z-positive": DependencyEdge(
                id="z-positive",
                expression="z >= 1",
                predicate="z >= 1",
                relation=DependencyRelation.BOUND,
                participants=("z",),
                drivers=(),
                dependents=("z",),
                status=DependencyStatus.CONFIRMED,
            ),
            "y-static-two": DependencyEdge(
                id="y-static-two",
                expression="y == 2",
                predicate="y == 2",
                relation=DependencyRelation.EQUALITY,
                participants=("y",),
                drivers=(),
                dependents=("y",),
                status=DependencyStatus.STATIC_CANDIDATE,
                confidence=0.95,
            ),
        },
    )
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    validator_path = tmp_path / "validator.json"
    validator_path.write_text(json.dumps({"command": ["validate"]}), encoding="utf-8")
    workloads_path = tmp_path / "workloads.yaml"
    workloads_path.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "toy",
                        "baseline_id": "toy-base",
                        "family": "dense",
                        "baseline_config": "baseline.json",
                        "dependency_graph": "graph.json",
                        "static_dependency_graph": "graph.json",
                        "native_validator_manifest": "validator.json",
                        "semantic_anchors": [],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    intents_source = tmp_path / "intents.yaml"
    intents_source.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "toy-intents",
                "metadata": {},
                "intents": [
                    {
                        "intent_id": "x-nine",
                        "workload_id": "toy",
                        "baseline_id": "toy-base",
                        "target_parameter": "x",
                        "target_value": 9,
                        "intent_class": "divisibility_adjacent_value",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    frozen = tmp_path / "intents.frozen.yaml"
    freeze_intent_file(intents_source, frozen)

    payload = plan_campaign(
        load_campaign_workloads(workloads_path),
        load_frozen_intents(frozen),
    )
    cases = {item["method"]: item for item in payload["cases"]}

    assert payload["case_count"] == 6
    assert cases["raw_mutation"]["assignments"] == {"x": 9}
    assert cases["native_validator_guided"]["preflight"] == "native_validator"
    assert cases["native_validator_guided"]["status"] == "ready"
    assert cases["constraint_filter_only"]["status"] == "filtered"
    assert set(cases["constraint_filter_only"]["violated_constraints"]) == {
        "xy",
        "z-positive",
    }
    local = cases["configfuzz"]
    assert local["status"] == "ready"
    assert local["assignments"]["x"] == 9
    assert "y" in local["coordinated_parameters"]
    assert "z" not in local["coordinated_parameters"]
    assert local["metadata"]["constraint_treatment"] == "status_and_confidence_aware"
    static_hard = cases["static_hard_configfuzz"]
    assert static_hard["status"] == "unsat"
    assert static_hard["metadata"]["constraint_treatment"] == "all_static_candidates_hard"
    assert static_hard["metadata"]["constraint_graph_source"] == "pre_validation_static_graph"
    global_repair = cases["global_repair"]
    assert global_repair["status"] == "ready"
    assert global_repair["assignments"]["x"] == 9
    assert set(global_repair["coordinated_parameters"]) == {"y", "z"}
    assert all(item["target_value_preserved"] is True for item in payload["cases"])


def test_native_validator_case_is_unknown_without_manifest(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"x": 1}), encoding="utf-8")
    graph = DependencyGraph(
        nodes={"x": DependencyNode("x", DependencyNodeKind.PARAMETER)},
        edges={},
    )
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    workloads_path = tmp_path / "workloads.yaml"
    workloads_path.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "w",
                        "baseline_id": "b",
                        "family": "dense",
                        "baseline_config": "baseline.json",
                        "dependency_graph": "graph.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    intent = MutationIntent(
        intent_id="x-two",
        workload_id="w",
        baseline_id="b",
        target_parameter="x",
        target_value=2,
        intent_class="boundary",
    )

    payload = plan_campaign(
        load_campaign_workloads(workloads_path),
        [intent],
        methods=[ExperimentMethod.NATIVE_VALIDATOR_GUIDED],
    )
    case = payload["cases"][0]

    assert case["status"] == "unknown"
    assert case["metadata"]["reason"] == "native validator manifest is not bound"


def test_frozen_intent_hash_is_verified(tmp_path: Path) -> None:
    source = tmp_path / "intents.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "metadata": {},
                "intents": [
                    {
                        "intent_id": "one",
                        "workload_id": "w",
                        "baseline_id": "b",
                        "target_parameter": "x",
                        "target_value": 1,
                        "intent_class": "boundary",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen.yaml"
    freeze_intent_file(source, frozen)
    payload = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    payload["intents"][0]["target_value"] = 2
    frozen.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_intents(frozen)


def test_campaign_rejects_baseline_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"x": 1}), encoding="utf-8")
    graph = DependencyGraph(
        nodes={"x": DependencyNode("x", DependencyNodeKind.PARAMETER)},
        edges={},
    )
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    workloads_path = tmp_path / "workloads.yaml"
    workloads_path.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "w",
                        "baseline_id": "expected",
                        "family": "dense",
                        "baseline_config": "baseline.json",
                        "dependency_graph": "graph.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    workloads = load_campaign_workloads(workloads_path)
    from configfuzz.experiment import MutationIntent

    intent = MutationIntent(
        intent_id="mismatch",
        workload_id="w",
        baseline_id="other",
        target_parameter="x",
        target_value=2,
        intent_class="boundary",
    )

    with pytest.raises(ValueError, match="does not match"):
        plan_campaign(workloads, [intent], methods=[ExperimentMethod.RAW_MUTATION])
