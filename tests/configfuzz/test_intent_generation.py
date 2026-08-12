from __future__ import annotations

import json
from pathlib import Path

import yaml

from configfuzz.intent_generation import generate_intent_payload, load_workloads


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus/lmsv/manual_constraints.yaml"


def test_generate_intents_from_bound_workload(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "model": {"hidden_size": 16},
                "parallel": {
                    "tensor_model_parallel_size": 2,
                    "pipeline_model_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                    "context_parallel_size": 1,
                    "sequence_parallel": False,
                },
                "world_size": 8,
                "bias_dropout_fusion": True,
            }
        ),
        encoding="utf-8",
    )
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "dense",
                        "family": "dense_transformer",
                        "baseline_id": "dense-base",
                        "baseline_config": "baseline.json",
                        "constraint_ids": [
                            "lmsv.task1.hidden-size-tp-divisibility",
                            "lmsv.task1.sequence-parallel-requires-tp",
                            "lmsv.task6.pool.bias_dropout_fusion.enum",
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    payload = generate_intent_payload(CORPUS, workloads)
    intents = payload["intents"]

    assert payload["metadata"]["workload_count"] == 1
    assert payload["metadata"]["intent_count"] == len(intents)
    assert any(
        item["target_parameter"] == "model.hidden_size"
        and item["target_value"] == 17
        and item["intent_class"] == "divisibility_adjacent_value"
        and item["intent_pool"] == "constraint_challenge"
        for item in intents
    )
    assert any(
        item["target_parameter"] == "model.hidden_size"
        and item["target_value"] == 17
        and item["intent_pool"] == "method_independent"
        and item["source_constraint_ids"] == []
        for item in intents
    )
    assert any(
        item["target_parameter"] == "parallel.sequence_parallel"
        and item["target_value"] is True
        and item["intent_pool"] == "constraint_challenge"
        for item in intents
    )
    assert any(
        item["target_parameter"] == "bias_dropout_fusion"
        and item["target_value"] is False
        and item["intent_class"] == "enumeration_alternative"
        and item["intent_pool"] == "constraint_challenge"
        for item in intents
    )
    assert any(
        item["target_parameter"] == "tensor_model_parallel_size"
        and item["target_value"] == 8
        and item["intent_class"] == "parallel_topology"
        and item["intent_pool"] == "method_independent"
        for item in intents
    )
    assert len({item["intent_id"] for item in intents}) == len(intents)
    assert len(
        {
            (
                item["workload_id"],
                item["intent_pool"],
                item["target_parameter"],
                json.dumps(item["target_value"], sort_keys=True),
            )
            for item in intents
        }
    ) == len(intents)


def test_unbound_workload_can_be_skipped(tmp_path: Path) -> None:
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "pending",
                        "family": "moe",
                        "baseline_id": "pending-base",
                        "baseline_config": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_workloads(workloads, skip_unbound=True) == []
    payload = generate_intent_payload(CORPUS, workloads, skip_unbound=True)
    assert payload["metadata"]["workload_count"] == 0
    assert payload["intents"] == []


def test_method_independent_pool_does_not_require_constraint_corpus(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"model": {"hidden_size": 16}, "world_size": 8}),
        encoding="utf-8",
    )
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "dense",
                        "family": "dense",
                        "baseline_id": "base",
                        "baseline_config": "baseline.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = generate_intent_payload(
        None, workloads, include_constraint_challenge=False
    )
    assert payload["metadata"]["source_corpus"] is None
    assert payload["metadata"]["constraint_challenge_included"] is False
    assert payload["intents"]
    assert all(item["intent_pool"] == "method_independent" for item in payload["intents"])
    assert all(not item["source_constraint_ids"] for item in payload["intents"])
