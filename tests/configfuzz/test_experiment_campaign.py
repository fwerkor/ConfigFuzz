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
    assert set(cases["constraint_filter_only"]["violated_constraints"]) == {"xy"}
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


def test_unconstrained_target_is_preserved_and_executed(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"model": {"hidden_size": 16}}), encoding="utf-8")
    graph = DependencyGraph(nodes={}, edges={})
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    validator_path = tmp_path / "validator.json"
    validator_path.write_text("{}", encoding="utf-8")
    workloads_path = tmp_path / "workloads.yaml"
    workloads_path.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "w",
                        "baseline_id": "b",
                        "baseline_config": "baseline.json",
                        "dependency_graph": "graph.json",
                        "static_dependency_graph": "graph.json",
                        "native_validator_manifest": "validator.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    intent = MutationIntent(
        intent_id="hidden-17",
        workload_id="w",
        baseline_id="b",
        target_parameter="model.hidden_size",
        target_value=17,
        intent_class="boundary",
    )

    payload = plan_campaign(
        load_campaign_workloads(workloads_path),
        [intent],
        methods=[
            ExperimentMethod.CONSTRAINT_FILTER_ONLY,
            ExperimentMethod.CONFIGFUZZ,
            ExperimentMethod.GLOBAL_REPAIR,
        ],
    )

    for case in payload["cases"]:
        assert case["status"] == "ready"
        assert case["assignments"] == {"model.hidden_size": 17}
        assert case["coordinated_parameters"] == []
        assert case["preflight"] == "no_applicable_recovered_constraint"


def test_target_placeholder_anchor_is_resolved_and_filter_ignores_unrelated_edges(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"x": 8, "y": 2, "unrelated": 1}), encoding="utf-8")
    graph = DependencyGraph(
        nodes={
            name: DependencyNode(name, DependencyNodeKind.PARAMETER)
            for name in ("x", "y", "unrelated", "missing")
        },
        edges={
            "xy": DependencyEdge(
                id="xy",
                expression="x % y == 0",
                predicate="x % y == 0",
                relation=DependencyRelation.DIVISIBILITY,
                participants=("x", "y"),
                drivers=("y",),
                dependents=("x",),
                status=DependencyStatus.CONFIRMED,
            ),
            "unrelated": DependencyEdge(
                id="unrelated",
                expression="unrelated < missing",
                predicate="unrelated < missing",
                relation=DependencyRelation.BOUND,
                participants=("unrelated", "missing"),
                drivers=(),
                dependents=("unrelated",),
                status=DependencyStatus.CONFIRMED,
            ),
        },
    )
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    validator = tmp_path / "validator.json"
    validator.write_text("{}", encoding="utf-8")
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "w",
                        "baseline_id": "b",
                        "baseline_config": "baseline.json",
                        "dependency_graph": "graph.json",
                        "static_dependency_graph": "graph.json",
                        "native_validator_manifest": "validator.json",
                        "semantic_anchors": ["target_parameter"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    intent = MutationIntent(
        intent_id="x-ten",
        workload_id="w",
        baseline_id="b",
        target_parameter="x",
        target_value=10,
        intent_class="boundary",
    )

    payload = plan_campaign(
        load_campaign_workloads(workloads),
        [intent],
        methods=[ExperimentMethod.CONSTRAINT_FILTER_ONLY, ExperimentMethod.CONFIGFUZZ],
    )
    cases = {case["method"]: case for case in payload["cases"]}

    assert cases["constraint_filter_only"]["status"] == "ready"
    assert cases["constraint_filter_only"]["unknown_constraints"] == []
    assert cases["configfuzz"]["status"] == "ready"
    assert "target_parameter" not in cases["configfuzz"]["metadata"]["missing_context"]


def test_missing_semantic_anchor_falls_back_to_target_only_execution(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"x": 1}), encoding="utf-8")
    graph = DependencyGraph(
        nodes={
            "x": DependencyNode("x", DependencyNodeKind.PARAMETER),
            "anchor": DependencyNode("anchor", DependencyNodeKind.PARAMETER),
        },
        edges={},
    )
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "w",
                        "baseline_id": "b",
                        "baseline_config": "baseline.json",
                        "dependency_graph": "graph.json",
                        "semantic_anchors": ["anchor"],
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
        load_campaign_workloads(workloads),
        [intent],
        methods=[ExperimentMethod.CONFIGFUZZ],
    )
    case = payload["cases"][0]

    assert case["status"] == "ready"
    assert case["assignments"] == {"x": 2}
    assert case["coordinated_parameters"] == []
    assert case["preflight"] == "manual_constraints_and_solver_fallback"
    assert case["solver_status"] == "unknown"
    assert case["metadata"]["fallback"] == "target_only_missing_semantic_anchor_context"


def test_campaign_runtime_context_is_available_to_solver(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "x": 1,
                "parallel": {
                    "pipeline_model_parallel_size": 1,
                    "tensor_model_parallel_size": 1,
                    "context_parallel_size": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    graph = DependencyGraph(
        nodes={
            "x": DependencyNode("x", DependencyNodeKind.PARAMETER),
            "parallel.pipeline_model_parallel_size": DependencyNode(
                "parallel.pipeline_model_parallel_size", DependencyNodeKind.PARAMETER
            ),
            "parallel.tensor_model_parallel_size": DependencyNode(
                "parallel.tensor_model_parallel_size", DependencyNodeKind.PARAMETER
            ),
            "parallel.context_parallel_size": DependencyNode(
                "parallel.context_parallel_size", DependencyNodeKind.PARAMETER
            ),
            "world_size": DependencyNode("world_size", DependencyNodeKind.ENVIRONMENT),
        },
        edges={
            "world": DependencyEdge(
                id="world",
                expression=(
                    "world_size % (parallel.tensor_model_parallel_size * "
                    "parallel.pipeline_model_parallel_size * parallel.context_parallel_size) == 0"
                ),
                predicate=(
                    "world_size % (parallel.tensor_model_parallel_size * "
                    "parallel.pipeline_model_parallel_size * parallel.context_parallel_size) == 0"
                ),
                relation=DependencyRelation.DIVISIBILITY,
                participants=(
                    "world_size",
                    "parallel.tensor_model_parallel_size",
                    "parallel.pipeline_model_parallel_size",
                    "parallel.context_parallel_size",
                ),
                drivers=("world_size",),
                dependents=("parallel.pipeline_model_parallel_size",),
                status=DependencyStatus.ENVIRONMENT_SPECIFIC,
            )
        },
    )
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "w",
                        "baseline_id": "b",
                        "baseline_config": "baseline.json",
                        "dependency_graph": "graph.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    intent = MutationIntent(
        intent_id="pp-three",
        workload_id="w",
        baseline_id="b",
        target_parameter="parallel.pipeline_model_parallel_size",
        target_value=3,
        intent_class="boundary",
    )

    payload = plan_campaign(
        load_campaign_workloads(workloads),
        [intent],
        methods=[ExperimentMethod.CONFIGFUZZ],
        runtime_context={"world_size": 2},
    )

    assert payload["runtime_context"] == {"world_size": 2}
    assert payload["cases"][0]["status"] == "unsat"

    workload_payload = yaml.safe_load(workloads.read_text(encoding="utf-8"))
    workload_payload["workloads"][0]["runtime_context"] = {"world_size": 2}
    workloads.write_text(yaml.safe_dump(workload_payload), encoding="utf-8")
    workload_context_payload = plan_campaign(
        load_campaign_workloads(workloads),
        [intent],
        methods=[ExperimentMethod.CONFIGFUZZ],
    )
    assert workload_context_payload["runtime_context"] == {}
    assert workload_context_payload["cases"][0]["status"] == "unsat"


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


def test_campaign_records_solver_timeout_and_elapsed_time(tmp_path: Path) -> None:
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
        methods=[ExperimentMethod.CONFIGFUZZ],
        solver_timeout_ms=250,
    )

    case = payload["cases"][0]
    assert payload["solver_timeout_ms"] == 250
    assert case["solver_timeout_ms"] == 250
    assert case["solver_seconds"] >= 0
    assert case["solver_seconds_recorded_at_runtime"] is False


def test_campaign_solver_timeout_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeout"):
        plan_campaign({}, [], solver_timeout_ms=0)
