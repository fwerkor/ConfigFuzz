from __future__ import annotations

from scripts.select_rq2_runtime_diversity_subset import select_subset
from scripts.summarize_rq2_gpu_results import _method_summary


def test_runtime_diversity_subset_selects_exact_hash_fraction_per_workload() -> None:
    cases = []
    for workload in ("w1", "w2"):
        for index in range(10):
            intent_id = f"{workload}-intent-{index}"
            for method in ("raw_mutation", "configfuzz"):
                cases.append(
                    {
                        "workload_id": workload,
                        "intent_id": intent_id,
                        "method": method,
                        "status": "ready",
                    }
                )
    plan = {"schema_version": 1, "cases": list(reversed(cases))}

    selected = select_subset(plan, 0.2)

    assert selected["intent_count"] == 4
    assert selected["case_count"] == 8
    assert selected["method_counts"] == {"configfuzz": 4, "raw_mutation": 4}
    counts = selected["metadata"]["runtime_diversity_subset"]["workloads"]
    assert counts["w1"]["selected_intents"] == 2
    assert counts["w2"]["selected_intents"] == 2


def test_method_summary_counts_explicit_runtime_diversity() -> None:
    rows = [
        {
            "generated": True,
            "target_value_preserved": True,
            "deepest_milestone": "completed",
            "outcome": "valid",
            "gpu_seconds": 1800.0,
            "coordinated_parameters": [],
            "runtime_branches": ["attention_mode=gqa"],
            "topologies": ["world=2,tp=1,pp=1,cp=1,ep=1"],
            "feature_interactions": ["attention"],
            "backend_paths": ["attention=sdpa"],
            "behavior_ids": [
                "branch:attention_mode=gqa",
                "topology:world=2,tp=1,pp=1,cp=1,ep=1",
                "feature:attention",
                "backend:attention=sdpa",
            ],
            "behavior_signature": "sig-a",
        },
        {
            "generated": True,
            "target_value_preserved": True,
            "deepest_milestone": "completed",
            "outcome": "valid",
            "gpu_seconds": 1800.0,
            "coordinated_parameters": [],
            "runtime_branches": ["attention_mode=mha"],
            "topologies": ["world=2,tp=1,pp=1,cp=1,ep=1"],
            "feature_interactions": ["attention", "moe"],
            "backend_paths": ["attention=sdpa"],
            "behavior_ids": [
                "branch:attention_mode=mha",
                "topology:world=2,tp=1,pp=1,cp=1,ep=1",
                "feature:attention",
                "feature:moe",
                "backend:attention=sdpa",
            ],
            "behavior_signature": "sig-b",
        },
    ]

    summary = _method_summary(rows)
    diversity = summary["diversity"]

    assert diversity["instrumented_execution_count"] == 2
    assert diversity["runtime_branches"] == 2
    assert diversity["topologies"] == 1
    assert diversity["feature_interactions"] == 2
    assert diversity["backend_paths"] == 1
    assert diversity["runtime_behavior_ids"] == 6
    assert diversity["behavior_signatures"] == 2
    assert diversity["behavior_signature_entropy_bits"] == 1.0
