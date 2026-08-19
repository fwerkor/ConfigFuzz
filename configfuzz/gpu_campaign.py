from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from configfuzz.dependencies import DependencyGraph
from configfuzz.feedback import FeedbackReport, apply_probe_feedback
from configfuzz.intervention_runner import InterventionExecutionManifest, run_intervention
from configfuzz.probing import ProbeSample
from configfuzz.selection import select_interventions


GPU_SUBJECTS: dict[str, str] = {
    "pytorch-native": "pytorch_v2.13.0.json",
    "deepspeed": "deepspeed_v0.19.1.json",
    "transformers-accelerate": "transformers_v5.9.0_accelerate_v1.14.0.json",
    "megatron-core": "megatron_core_v0.18.2.json",
}


GPU_HARNESS_FILES: dict[str, tuple[str, ...]] = {
    "pytorch-native": (
        "experiments/gpu/launch_pytorch_native.sh",
        "experiments/gpu/qualification/pytorch_native.py",
        "experiments/gpu/qualification/common.py",
    ),
    "deepspeed": (
        "experiments/gpu/launch_deepspeed.sh",
        "experiments/gpu/qualification/deepspeed_runner.py",
        "experiments/gpu/qualification/common.py",
    ),
    "transformers-accelerate": (
        "experiments/gpu/launch_transformers_accelerate.sh",
        "experiments/gpu/qualification/transformers_accelerate.py",
        "experiments/gpu/qualification/common.py",
    ),
    "megatron-core": (
        "experiments/gpu/launch_megatron_core.sh",
        "experiments/gpu/qualification/megatron_core.py",
        "experiments/gpu/qualification/common.py",
    ),
}

MEGATRON_PROCESS_TOPOLOGY_FIELDS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
)


def _keeps_qualified_process_topology(candidate: Any, baseline: Mapping[str, Any]) -> bool:
    """Require positive-control and repaired arms to use the qualified layout."""
    cases = [candidate.intervention.satisfying, candidate.intervention.repaired]
    for case in cases:
        if case is None:
            return False
        assignment = case.assignment
        for name in MEGATRON_PROCESS_TOPOLOGY_FIELDS:
            if name in assignment and name in baseline and assignment[name] != baseline[name]:
                return False
    return True

def build_frozen_gpu_targets(
    root: Path,
    *,
    limit_per_subject: int = 6,
    solver_timeout_ms: int = 3000,
) -> dict[str, Any]:
    if limit_per_subject <= 0:
        raise ValueError("limit_per_subject must be positive")
    if solver_timeout_ms <= 0:
        raise ValueError("solver_timeout_ms must be positive")

    subjects: list[dict[str, Any]] = []
    target_count = 0
    for subject, artifact_name in GPU_SUBJECTS.items():
        artifact = root / "artifacts" / "frameworks" / artifact_name
        manifest_path = root / "experiments" / "gpu" / "manifests" / f"{subject}.json"
        manifest = InterventionExecutionManifest.from_path(manifest_path)
        baseline = json.loads(manifest.baseline_config.read_text(encoding="utf-8"))
        graph_payload = json.loads(artifact.read_text(encoding="utf-8"))
        queue = select_interventions(
            DependencyGraph.from_dict(graph_payload),
            baseline,
            limit=limit_per_subject,
            solver_timeout_ms=solver_timeout_ms,
        )
        candidates = list(queue.candidates)
        skipped_unqualified_topology = 0
        if subject == "megatron-core":
            filtered = [
                candidate
                for candidate in candidates
                if _keeps_qualified_process_topology(candidate, baseline)
            ]
            skipped_unqualified_topology = len(candidates) - len(filtered)
            candidates = filtered
        targets = [
            {
                "rank": rank,
                "edge_id": candidate.edge_id,
                "expression": candidate.expression,
                "relation": candidate.relation.value,
                "static_status": candidate.status.value,
                "static_confidence": candidate.confidence,
                "selection_score": candidate.score,
                "score_components": dict(candidate.score_components),
                "intervention": candidate.intervention.to_dict(),
            }
            for rank, candidate in enumerate(candidates, start=1)
        ]
        selection_summary = dict(queue.to_dict()["summary"])
        selection_summary["pre_topology_filter_candidates"] = int(
            selection_summary["selected_candidates"]
        )
        selection_summary["skipped_unqualified_topology"] = skipped_unqualified_topology
        selection_summary["selected_candidates"] = len(targets)
        target_count += len(targets)
        subjects.append(
            {
                "subject": subject,
                "artifact": str(artifact.relative_to(root)),
                "manifest": str(manifest_path.relative_to(root)),
                "baseline": str(manifest.baseline_config.relative_to(root)),
                "artifact_sha256": _sha256_file(artifact),
                "manifest_sha256": _sha256_file(manifest_path),
                "baseline_sha256": _sha256_file(manifest.baseline_config),
                "harness_files": [
                    {
                        "path": path,
                        "sha256": _sha256_file(root / path),
                    }
                    for path in GPU_HARNESS_FILES[subject]
                ],
                "selection_summary": selection_summary,
                "targets": targets,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 2,
        "name": "gpu-cross-framework-validation-targets",
        "metadata": {
            "selection_basis": "static graph plus qualified baseline only",
            "limit_per_subject": limit_per_subject,
            "solver_timeout_ms": solver_timeout_ms,
            "subject_count": len(subjects),
            "target_count": target_count,
        },
        "subjects": subjects,
    }
    payload["frozen"] = {
        "sha256": _payload_sha256(payload),
        "target_count": target_count,
    }
    return payload


def load_frozen_gpu_targets(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("frozen GPU target file must contain an object")
    frozen = raw.get("frozen")
    if not isinstance(frozen, Mapping) or not isinstance(frozen.get("sha256"), str):
        raise ValueError("frozen GPU target file is missing its hash")
    expected = str(frozen["sha256"])
    actual = _payload_sha256(raw)
    if actual != expected:
        raise ValueError(f"frozen GPU target hash mismatch: expected {expected}, got {actual}")
    subjects = raw.get("subjects")
    if not isinstance(subjects, list):
        raise ValueError("frozen GPU target file is missing subjects")
    target_count = sum(
        len(item.get("targets", ()))
        for item in subjects
        if isinstance(item, Mapping)
    )
    if int(frozen.get("target_count", -1)) != target_count:
        raise ValueError("frozen GPU target count mismatch")
    return raw


def run_frozen_gpu_subject(
    root: Path,
    frozen: Mapping[str, Any],
    subject: str,
) -> dict[str, Any]:
    subject_payload = next(
        (
            item
            for item in frozen.get("subjects", ())
            if isinstance(item, Mapping) and item.get("subject") == subject
        ),
        None,
    )
    if subject_payload is None:
        raise KeyError(f"unknown frozen GPU subject: {subject}")

    artifact = root / str(subject_payload["artifact"])
    manifest_path = root / str(subject_payload["manifest"])
    baseline_path = root / str(subject_payload["baseline"])
    for path, key in (
        (artifact, "artifact_sha256"),
        (manifest_path, "manifest_sha256"),
        (baseline_path, "baseline_sha256"),
    ):
        expected = str(subject_payload[key])
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"{subject} {key} mismatch: expected {expected}, got {actual}")
    raw_harness_files = subject_payload.get("harness_files")
    if not isinstance(raw_harness_files, list) or not raw_harness_files:
        raise ValueError(f"{subject} frozen target is missing harness file hashes")
    for item in raw_harness_files:
        if not isinstance(item, Mapping):
            raise ValueError(f"{subject} harness file entry is malformed")
        harness_path = root / str(item["path"])
        expected = str(item["sha256"])
        actual = _sha256_file(harness_path)
        if actual != expected:
            raise ValueError(
                f"{subject} harness hash mismatch for {harness_path}: "
                f"expected {expected}, got {actual}"
            )

    graph_payload = json.loads(artifact.read_text(encoding="utf-8"))
    manifest = InterventionExecutionManifest.from_path(manifest_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rounds: list[dict[str, Any]] = []
    round_samples: list[list[ProbeSample]] = []
    outcome_counts: Counter[str] = Counter()
    for index, target in enumerate(subject_payload.get("targets", ()), start=1):
        if not isinstance(target, Mapping) or not isinstance(target.get("intervention"), Mapping):
            raise ValueError(f"{subject} target {index} is malformed")
        samples = run_intervention({"intervention": target["intervention"]}, manifest)
        round_samples.append(samples)
        outcome_counts.update(sample.outcome.label.value for sample in samples)
        rounds.append(
            {
                "index": index,
                "edge_id": str(target["edge_id"]),
                "expression": str(target["expression"]),
                "selection_rank": int(target["rank"]),
                "samples": [sample.to_dict() for sample in samples],
            }
        )

    graph, independent_feedback, aggregate_feedback = apply_frozen_feedback(
        graph_payload,
        baseline,
        round_samples,
    )
    for round_payload, feedback in zip(rounds, independent_feedback, strict=True):
        round_payload["feedback"] = feedback.to_dict()

    status_counts = Counter(edge.status.value for edge in graph.edges.values())
    return {
        "schema_version": 1,
        "subject": subject,
        "frozen_targets_sha256": frozen["frozen"]["sha256"],
        "summary": {
            "targets": len(rounds),
            "samples": sum(len(item["samples"]) for item in rounds),
            "outcomes": dict(sorted(outcome_counts.items())),
            "edge_statuses": dict(sorted(status_counts.items())),
            "paired_interventions": aggregate_feedback.paired_interventions,
        },
        "aggregate_feedback": aggregate_feedback.to_dict(),
        "rounds": rounds,
        "dependency_graph": graph.to_dict(),
    }



def apply_frozen_feedback(
    graph_payload: Mapping[str, Any],
    baseline: Mapping[str, Any],
    round_samples: list[list[ProbeSample]],
) -> tuple[DependencyGraph, list[FeedbackReport], FeedbackReport]:
    """Evaluate frozen targets without letting target order change hard constraints."""
    independent_feedback = []
    all_samples: list[ProbeSample] = []
    for samples in round_samples:
        target_graph = DependencyGraph.from_dict(graph_payload)
        independent_feedback.append(
            apply_probe_feedback(target_graph, baseline, samples)
        )
        all_samples.extend(samples)

    aggregate_graph = DependencyGraph.from_dict(graph_payload)
    aggregate_feedback = apply_probe_feedback(
        aggregate_graph,
        baseline,
        all_samples,
    )
    return aggregate_graph, independent_feedback, aggregate_feedback

def summarize_frozen_gpu_results(
    frozen: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    *,
    hardware: Mapping[str, Any],
    campaign_date: str,
    runner_revision: str,
    result_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic cross-framework summary from frozen-campaign outputs."""
    expected_subjects = [
        str(item["subject"])
        for item in frozen.get("subjects", ())
        if isinstance(item, Mapping) and item.get("subject") is not None
    ]
    if not expected_subjects:
        raise ValueError("frozen GPU target payload has no subjects")

    subjects: list[dict[str, Any]] = []
    aggregate_outcomes: Counter[str] = Counter()
    aggregate_targets = 0
    aggregate_samples = 0
    aggregate_confirmed = 0
    aggregate_scope_disputed = 0
    aggregate_unresolved = 0
    frozen_sha = str(frozen.get("frozen", {}).get("sha256", ""))

    for subject in expected_subjects:
        result = results.get(subject)
        if not isinstance(result, Mapping):
            raise ValueError(f"missing frozen GPU result for {subject}")
        if str(result.get("subject")) != subject:
            raise ValueError(f"GPU result subject mismatch for {subject}")
        if str(result.get("frozen_targets_sha256")) != frozen_sha:
            raise ValueError(f"GPU result frozen-target hash mismatch for {subject}")

        confirmed: list[str] = []
        scope_disputed: list[str] = []
        unresolved: list[str] = []
        target_detail: list[dict[str, Any]] = []
        outcome_counts: Counter[str] = Counter()
        rounds = result.get("rounds", ())
        if not isinstance(rounds, list):
            raise ValueError(f"GPU result rounds must be a list for {subject}")
        for round_payload in rounds:
            if not isinstance(round_payload, Mapping):
                raise ValueError(f"GPU result round is malformed for {subject}")
            edge_id = str(round_payload["edge_id"])
            feedback = round_payload.get("feedback")
            if not isinstance(feedback, Mapping):
                raise ValueError(f"GPU result feedback is missing for {subject}:{edge_id}")
            paired = int(feedback.get("paired_interventions", 0))
            disputed_edges = {str(item) for item in feedback.get("scope_disputed_edges", ())}
            if paired > 0:
                status = "paired_confirmed"
                confirmed.append(edge_id)
            elif edge_id in disputed_edges:
                status = "scope_disputed"
                scope_disputed.append(edge_id)
            else:
                status = "unresolved"
                unresolved.append(edge_id)

            samples = round_payload.get("samples", ())
            if not isinstance(samples, list):
                raise ValueError(f"GPU result samples must be a list for {subject}:{edge_id}")
            round_outcomes: list[str] = []
            for sample in samples:
                if not isinstance(sample, Mapping):
                    raise ValueError(f"GPU result sample is malformed for {subject}:{edge_id}")
                outcome = sample.get("outcome")
                if not isinstance(outcome, Mapping) or outcome.get("label") is None:
                    raise ValueError(f"GPU result sample outcome is malformed for {subject}:{edge_id}")
                label = str(outcome["label"])
                round_outcomes.append(label)
                outcome_counts[label] += 1
            target_detail.append(
                {
                    "edge_id": edge_id,
                    "expression": str(round_payload["expression"]),
                    "status": status,
                    "outcomes": round_outcomes,
                }
            )

        aggregate_outcomes.update(outcome_counts)
        aggregate_targets += len(rounds)
        aggregate_samples += sum(outcome_counts.values())
        aggregate_confirmed += len(confirmed)
        aggregate_scope_disputed += len(scope_disputed)
        aggregate_unresolved += len(unresolved)
        subject_summary = {
            "subject": subject,
            "targets": len(rounds),
            "samples": sum(outcome_counts.values()),
            "paired_confirmed": len(confirmed),
            "scope_disputed": len(scope_disputed),
            "unresolved": len(unresolved),
            "outcomes": dict(sorted(outcome_counts.items())),
            "confirmed_edge_ids": confirmed,
            "scope_disputed_edge_ids": scope_disputed,
            "targets_detail": target_detail,
        }
        if result_hashes is not None and subject in result_hashes:
            subject_summary["result_sha256"] = str(result_hashes[subject])
        subjects.append(subject_summary)

        # Aggregate accounting above is independent of optional integrity metadata.


    expected_target_count = int(frozen.get("frozen", {}).get("target_count", -1))
    if aggregate_targets != expected_target_count:
        raise ValueError(
            f"formal GPU result target count mismatch: expected {expected_target_count}, "
            f"got {aggregate_targets}"
        )

    return {
        "schema_version": 2,
        "name": "gpu-cross-framework-frozen-validation-summary",
        "campaign_date": campaign_date,
        "runner_revision": runner_revision,
        "frozen_targets_sha256": frozen_sha,
        "frozen_target_count": expected_target_count,
        "protocol": {
            "selection_basis": (
                "versioned static graph plus qualified effective baseline; "
                "no runtime outcomes used for target selection"
            ),
            "scope_filtering": (
                "execution-stage scope is matched before intervention selection and feedback"
            ),
            "feedback_application": (
                "independent per target plus order-independent aggregate feedback"
            ),
            "harness_integrity": (
                "artifact, baseline, manifest, launch script, qualification runner, "
                "and shared runner source are SHA-256 pinned; ConfigFuzz runner revision is recorded"
            ),
            "roles": ["satisfying", "violating", "repaired"],
        },
        "hardware": {
            "accelerator": hardware.get("accelerator"),
            "device_count": hardware.get("device_count"),
            "distributed_backend": hardware.get("distributed_backend"),
        },
        "aggregate": {
            "targets": aggregate_targets,
            "samples": aggregate_samples,
            "paired_confirmed": aggregate_confirmed,
            "scope_disputed": aggregate_scope_disputed,
            "unresolved": aggregate_unresolved,
            "outcomes": dict(sorted(aggregate_outcomes.items())),
        },
        "subjects": subjects,
    }



def dump_frozen_gpu_targets(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical_payload = {key: value for key, value in payload.items() if key != "frozen"}
    canonical = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
