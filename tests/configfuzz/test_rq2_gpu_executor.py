from __future__ import annotations

import json
from pathlib import Path

import yaml

from configfuzz.experiment import ExecutionMilestone, ExperimentOutcome
from configfuzz.experiment_campaign import load_campaign_workloads
from configfuzz.rq2_gpu_executor import (
    behavior_signature,
    classify_outcome,
    deepest_milestone,
    materialize_profile,
)


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
