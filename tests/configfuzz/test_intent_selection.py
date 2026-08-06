from __future__ import annotations

import json
from pathlib import Path

import yaml

from configfuzz.intent_selection import select_balanced_intents


def test_balanced_selection_is_unique_and_parameter_diverse(tmp_path: Path) -> None:
    intents = []
    for parameter in (
        "model.hidden_size",
        "model.num_layers",
        "training.global_batch_size",
    ):
        for value in range(1, 8):
            intents.append(
                {
                    "intent_id": f"w-{parameter}-{value}",
                    "workload_id": "w",
                    "baseline_id": "base",
                    "target_parameter": parameter,
                    "target_value": value,
                    "intent_class": (
                        "repair_boundary" if value == 1 else "integer_scale_boundary"
                    ),
                    "source_constraint_ids": [f"constraint-{parameter}"],
                    "metadata": {},
                }
            )
    candidate_file = tmp_path / "candidates.yaml"
    candidate_file.write_text(
        yaml.safe_dump({"intents": intents}, sort_keys=False), encoding="utf-8"
    )
    registry_file = tmp_path / "workloads.yaml"
    registry_file.write_text(
        yaml.safe_dump(
            {
                "workloads": [
                    {
                        "workload_id": "w",
                        "metadata": {"priority": "primary", "minimum_intents": 12},
                    },
                    {
                        "workload_id": "fallback",
                        "metadata": {"priority": "fallback", "minimum_intents": 0},
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    payload = select_balanced_intents(candidate_file, registry_file)
    selected = payload["intents"]

    assert len(selected) == 12
    assert {item["workload_id"] for item in selected} == {"w"}
    assert {item["target_parameter"] for item in selected} == {
        "model.hidden_size",
        "model.num_layers",
        "training.global_batch_size",
    }
    target_keys = {
        (
            item["workload_id"],
            item["target_parameter"],
            json.dumps(item["target_value"], sort_keys=True),
        )
        for item in selected
    }
    assert len(target_keys) == len(selected)
    assert payload["metadata"]["workload_selection"]["w"]["selected_count"] == 12
