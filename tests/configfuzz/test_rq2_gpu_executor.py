from __future__ import annotations

import json
import socket
from pathlib import Path

import yaml

from configfuzz.experiment import ExecutionMilestone, ExperimentOutcome
from configfuzz.experiment_campaign import load_campaign_workloads
from configfuzz.rq2_gpu_executor import (
    _available_master_port,
    behavior_signature,
    classify_outcome,
    deepest_milestone,
    materialize_profile,
    parse_runtime_events,
)


def test_available_master_port_skips_an_occupied_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        assert _available_master_port(port) != port


def test_materialize_profile_binds_exact_and_unique_leaf_assignments(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"model": {"hidden_size": 16}, "training": {"learning_rate": 1e-4}}),
        encoding="utf-8",
    )
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({"nodes": {}, "edges": {}}), encoding="utf-8")
    registry = tmp_path / "workloads.yaml"
    registry.write_text(
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
    workload = load_campaign_workloads(registry)["w"]
    profile = materialize_profile(
        workload,
        {"assignments": {"hidden_size": 32, "training.learning_rate": 2e-4}},
    )

    assert profile["model"]["hidden_size"] == 32
    assert profile["training"]["learning_rate"] == 2e-4


def test_gpu_output_classification_uses_deepest_reported_milestone() -> None:
    output = "\n".join(
        [
            "CONFIGFUZZ_MILESTONE:argument_parsing",
            "CONFIGFUZZ_MILESTONE:model_construction",
            "CONFIGFUZZ_MILESTONE:forward",
            "RuntimeError: kernel failed",
        ]
    )
    milestone = deepest_milestone(output, False)

    assert milestone is ExecutionMilestone.FORWARD
    assert classify_outcome(output, 1, False, milestone) is ExperimentOutcome.UNEXPLAINED_FAILURE
    assert behavior_signature(output, milestone, ExperimentOutcome.UNEXPLAINED_FAILURE)


def test_runtime_events_are_categorized_and_signed_independently_of_outcome() -> None:
    output = "\n".join(
        [
            'CONFIGFUZZ_RUNTIME_EVENT:{"kind":"branch","value":"attention_mode=gqa"}',
            'CONFIGFUZZ_RUNTIME_EVENT:{"kind":"backend","value":"attention=sdpa"}',
            'CONFIGFUZZ_RUNTIME_EVENT:{"kind":"topology","value":"world=2,tp=1,pp=1,cp=1,ep=1"}',
            'CONFIGFUZZ_RUNTIME_EVENT:{"kind":"feature","value":"attention"}',
            'CONFIGFUZZ_RUNTIME_EVENT:{"kind":"feature","value":"attention"}',
            "RuntimeError: unrelated failure text",
        ]
    )

    events = parse_runtime_events(output)

    assert events["branch"] == ("attention_mode=gqa",)
    assert events["backend"] == ("attention=sdpa",)
    assert events["topology"] == ("world=2,tp=1,pp=1,cp=1,ep=1",)
    assert events["feature"] == ("attention",)
    assert events["behavior_ids"] == (
        "backend:attention=sdpa",
        "branch:attention_mode=gqa",
        "feature:attention",
        "topology:world=2,tp=1,pp=1,cp=1,ep=1",
    )
    assert events["behavior_signature"] is not None


def test_explicit_configuration_error_after_distributed_init_is_expected_rejection() -> None:
    output = "\n".join(
        [
            "CONFIGFUZZ_MILESTONE:configuration_validation",
            "CONFIGFUZZ_MILESTONE:distributed_initialization",
            "ValueError: Only one of fp16 and bf16 should be True.",
        ]
    )
    milestone = deepest_milestone(output, False)

    assert milestone is ExecutionMilestone.PROCESS_GROUP_INITIALIZATION
    assert classify_outcome(output, 1, False, milestone) is ExperimentOutcome.EXPECTED_REJECTION
