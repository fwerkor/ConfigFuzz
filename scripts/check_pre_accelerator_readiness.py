#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Check every ConfigFuzz prerequisite that does not require an accelerator.")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "readiness.pre_accelerator.json")
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            blockers.append(name)

    protocol = load_yaml(ROOT / "experiments/protocol.yaml")
    replication = protocol.get("rq2", {}).get("replication_policy", {})
    check(
        "rq2_replication_policy_frozen",
        replication.get("primary_seed") == 2026
        and replication.get("seed_sensitivity", {}).get("seed_count") == 5
        and "minimum_random_seeds" not in protocol.get("rq2", {}),
        replication,
    )

    audit = load_yaml(ROOT / "experiments/rq1/constraint_audit.yaml")
    check("rq1_audit_has_93_constraints", len(audit.get("records", [])) == 93, len(audit.get("records", [])))

    baseline_dir = ROOT / "experiments/rq2/baselines/canonical-v1"
    baselines = sorted(baseline_dir.glob("*.json"))
    scale_ok = True
    workload_ids = []
    for path in baselines:
        item = load_json(path)
        workload_ids.append(str(item.get("workload_id")))
        model = item.get("model", {})
        scale_ok &= (
            item.get("model_scale_profile") == "npu-reduced-26.1"
            and model.get("hidden_size") == 512
            and model.get("ffn_hidden_size") == 1024
            and model.get("num_layers") == 4
            and model.get("seq_length") == 128
        )
    check("rq2_canonical_workloads_complete", len(baselines) == 7 and len(set(workload_ids)) == 7 and scale_ok, workload_ids)

    model_preflight = load_json(ROOT / "experiments/rq2/model_preflight.prequalified.json")
    check(
        "rq2_all_model_families_pass_cpu_forward_backward",
        model_preflight.get("passed") == 7
        and model_preflight.get("failed") == 0
        and all(row.get("status") == "model_forward_backward_passed" for row in model_preflight.get("records", [])),
        {"passed": model_preflight.get("passed"), "failed": model_preflight.get("failed")},
    )

    intents_path = ROOT / "experiments/rq2/intents.prequalified.frozen.yaml"
    intents = load_yaml(intents_path)
    intent_rows = intents.get("intents", [])
    by_workload = Counter(str(row.get("workload_id")) for row in intent_rows)
    check(
        "rq2_frozen_intents_1050_balanced",
        len(intent_rows) == 1050 and len(by_workload) == 7 and set(by_workload.values()) == {150},
        {"count": len(intent_rows), "by_workload": dict(sorted(by_workload.items())), "sha256": hashlib.sha256(intents_path.read_bytes()).hexdigest()},
    )

    matrix = load_yaml(ROOT / "experiments/rq2/framework_workload_matrix.yaml")
    bindings = matrix.get("bindings", [])
    formal = [row for row in bindings if row.get("formal_rq2")]
    frameworks = {str(row.get("framework_id")) for row in bindings}
    check("rq2_framework_workload_matrix_frozen", len(frameworks) == 6 and len(formal) == 38, {"frameworks": sorted(frameworks), "formal_pairs": len(formal)})

    runtime = load_yaml(ROOT / "experiments/rq2/runtime_subjects.prequalified.yaml")
    runtime_subjects = {str(row["framework_id"]): row for row in runtime.get("subjects", [])}
    expected_supported: dict[str, set[str]] = defaultdict(set)
    for row in formal:
        expected_supported[str(row["framework_id"])].add(str(row["workload_id"]))
    runtime_ok = set(runtime_subjects) == frameworks
    runtime_details = {}
    for framework in sorted(expected_supported):
        actual = set(runtime_subjects.get(framework, {}).get("supported_workloads", []))
        runtime_ok &= actual == expected_supported[framework]
        runtime_details[framework] = sorted(actual)
    check("rq2_runtime_subjects_match_matrix", runtime_ok, runtime_details)

    referenced_paths = [
        ROOT / "experiments/gpu/launch_rq2_pytorch.sh",
        ROOT / "experiments/gpu/launch_rq2_deepspeed.sh",
        ROOT / "experiments/gpu/launch_rq2_accelerate.sh",
        ROOT / "experiments/gpu/launch_rq2_megatron.sh",
        ROOT / "experiments/gpu/rq2_family_runner.py",
        ROOT / "experiments/gpu/rq2_megatron_runner.py",
        ROOT / "experiments/runtime/mindspeed_persistent_worker_26_1.py",
        ROOT / "experiments/runtime/mindspeed_26_1_persistent.sh",
        ROOT / "experiments/runtime/benchmark_rq1_persistent_26_1.py",
        ROOT / "scripts/run_rq2_campaign.py",
        ROOT / "scripts/prepare_rq2_campaign_inputs.py",
    ]
    check("accelerator_launch_harnesses_present", all(path.exists() for path in referenced_paths), [str(path.relative_to(ROOT)) for path in referenced_paths if not path.exists()])

    schedule = load_yaml(ROOT / "experiments/rq2/replication_summary.prequalified.yaml")
    subject_summaries = schedule.get("subjects", [])
    check(
        "rq2_replication_schedule_frozen",
        schedule.get("total_record_count") == 58560 and len(subject_summaries) == 6,
        {"total_record_count": schedule.get("total_record_count"), "subject_count": len(subject_summaries)},
    )

    rq3_bindings = load_yaml(ROOT / "experiments/rq3/search_bindings.prequalified.yaml")
    rq3_cases = rq3_bindings.get("cases", [])
    check(
        "rq3_historical_search_cases_bound",
        len(rq3_cases) == 25
        and rq3_bindings.get("metadata", {}).get("source_commit_candidates") == 23
        and all(row.get("status") == "prepared_pending_accelerator_replay" for row in rq3_cases),
        {"source_candidates": rq3_bindings.get("metadata", {}).get("source_commit_candidates"), "logical_root_causes": len(rq3_cases)},
    )

    history_preflight_path = ROOT / "experiments/rq3/history_preflight.prequalified.json"
    history = load_json(history_preflight_path) if history_preflight_path.exists() else {}
    check(
        "rq3_historical_commits_and_harness_blobs_available",
        history.get("logical_case_count") == 25 and history.get("ready_count") == 25 and history.get("failed_count") == 0,
        {"logical_case_count": history.get("logical_case_count"), "ready_count": history.get("ready_count"), "failed_count": history.get("failed_count")},
    )

    budgets = protocol.get("rq3", {}).get("search_budgets", {})
    check(
        "rq3_search_budgets_frozen",
        budgets.get("historical_replay", {}).get("per_bug_per_method_max_generated_tests") == 200
        and budgets.get("current_version", {}).get("per_framework_per_method_max_generated_tests") == 2000,
        budgets,
    )

    frozen_gpu = load_yaml(ROOT / "experiments/gpu/validation_targets.frozen.yaml")
    gpu_summary = load_yaml(ROOT / "experiments/gpu/formal_results_summary.yaml")
    current_hash = str(frozen_gpu.get("frozen", {}).get("sha256", ""))
    stored_hash = str(gpu_summary.get("frozen_targets_sha256", ""))
    gpu_aggregate = gpu_summary.get("aggregate", {})
    gpu_summary_complete = (
        current_hash == stored_hash
        and gpu_summary.get("campaign_date") == "2026-08-17"
        and gpu_summary.get("frozen_target_count") == 19
        and gpu_aggregate.get("targets") == 19
        and gpu_aggregate.get("samples") == 57
        and gpu_aggregate.get("paired_confirmed") == 15
        and gpu_aggregate.get("scope_disputed") == 2
        and gpu_aggregate.get("unresolved") == 2
    )
    check(
        "gpu_rq1_frozen_summary_matches_targets",
        gpu_summary_complete,
        {
            "campaign_date": gpu_summary.get("campaign_date"),
            "current_targets_sha256": current_hash,
            "stored_summary_targets_sha256": stored_hash,
            "targets": gpu_aggregate.get("targets"),
            "samples": gpu_aggregate.get("samples"),
            "paired_confirmed": gpu_aggregate.get("paired_confirmed"),
            "scope_disputed": gpu_aggregate.get("scope_disputed"),
            "unresolved": gpu_aggregate.get("unresolved"),
        },
    )
    accelerator_gates = [
        "Run the NPU persistent-worker A/B benchmark on an idle 910B card.",
        "Qualify the remaining 14 Ascend PTA/MSA framework-workload pairs to checkpoint_save_load and promote their bindings.",
        "Refreeze the 1050 intents only if remaining Ascend qualification changes a qualified baseline schema.",
        "Execute formal Ascend RQ1 satisfying/violating pairs and collect failure-stage/cost metrics.",
        "Execute the frozen RQ2 schedule; FILTERED/UNSAT records remain accelerator-free.",
        "Execute the 25 logical RQ3 historical searches plus fixed-revision confirmation, then run current-version discovery under the frozen budgets.",
    ]

    report = {
        "schema_version": 1,
        "name": "configfuzz-pre-accelerator-readiness",
        "status": "ready_for_accelerator" if not blockers else "blocked_before_accelerator",
        "non_accelerator_blockers": blockers,
        "checks": checks,
        "accelerator_gates": accelerator_gates,
        "external_followups": [
            "Complete independent secondary RQ1 source review before paper claims are finalized.",
            "Seek developer confirmation for newly discovered RQ3 current-version bugs after reproduction/minimization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "blocker_count": len(blockers), "accelerator_gate_count": len(accelerator_gates)}, sort_keys=True))
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
