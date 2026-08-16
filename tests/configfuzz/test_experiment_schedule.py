from __future__ import annotations

from configfuzz.experiment import ExperimentMethod, MutationIntent
from configfuzz.experiment_schedule import (
    PRIMARY_SEED,
    SENSITIVITY_SEEDS,
    build_replication_schedule,
    is_seed_sensitivity_intent,
    summarize_schedule,
)


def _intent(intent_id: str, workload_id: str = "qwen2-train") -> MutationIntent:
    return MutationIntent(
        intent_id=intent_id,
        workload_id=workload_id,
        baseline_id=f"{workload_id}-canonical-v1",
        target_parameter="model.hidden_size",
        target_value=768,
        intent_class="integer_boundary",
    )


def test_seed_sensitivity_selection_is_deterministic_and_approximately_one_fifth() -> None:
    selected_a = [f"intent-{index}" for index in range(1000) if is_seed_sensitivity_intent(f"intent-{index}")]
    selected_b = [f"intent-{index}" for index in range(1000) if is_seed_sensitivity_intent(f"intent-{index}")]
    assert selected_a == selected_b
    assert 160 <= len(selected_a) <= 240


def test_schedule_has_one_primary_and_four_extra_seeds_for_selected_intent() -> None:
    selected_id = next(f"intent-{index}" for index in range(1000) if is_seed_sensitivity_intent(f"intent-{index}"))
    methods = (ExperimentMethod.RAW_MUTATION, ExperimentMethod.CONFIGFUZZ)
    schedule = build_replication_schedule([_intent(selected_id)], methods=methods)

    assert len(schedule) == len(methods) * len(SENSITIVITY_SEEDS)
    for method in methods:
        rows = [row for row in schedule if row.method == method]
        assert {row.seed for row in rows} == set(SENSITIVITY_SEEDS)
        assert sum(row.role == "primary" and row.seed == PRIMARY_SEED for row in rows) == 1
        assert sum(row.role == "seed_sensitivity" for row in rows) == 4


def test_schedule_filters_unsupported_workloads_and_summarizes() -> None:
    intents = [_intent("a", "qwen2-train"), _intent("b", "internvl3-train")]
    methods = (ExperimentMethod.CONFIGFUZZ,)
    schedule = build_replication_schedule(intents, methods=methods, supported_workloads={"qwen2-train"})
    summary = summarize_schedule(schedule)

    assert {row.workload_id for row in schedule} == {"qwen2-train"}
    assert summary["primary_intent_count"] == 1
    assert summary["method_counts"] == {"configfuzz": len(schedule)}
