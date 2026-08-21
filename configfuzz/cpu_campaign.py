from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from configfuzz.dependencies import DependencyGraph
from configfuzz.gpu_campaign import (
    load_frozen_gpu_targets,
    run_frozen_gpu_subject,
    summarize_frozen_gpu_results,
)
from configfuzz.intervention_runner import InterventionExecutionManifest
from configfuzz.selection import select_interventions


CPU_SUBJECTS: dict[str, str] = {
    "pytorch-native": "pytorch_v2.13.0.json",
    "deepspeed": "deepspeed_v0.19.1.json",
    "transformers-accelerate": "transformers_v5.9.0_accelerate_v1.14.0.json",
}

CPU_HARNESS_FILES: dict[str, tuple[str, ...]] = {
    "pytorch-native": (
        "experiments/cpu/launch_pytorch_native.sh",
        "experiments/cpu/qualification/pytorch_native.py",
        "experiments/cpu/qualification/common.py",
    ),
    "deepspeed": (
        "experiments/cpu/launch_deepspeed.sh",
        "experiments/cpu/qualification/deepspeed_runner.py",
        "experiments/cpu/qualification/common.py",
    ),
    "transformers-accelerate": (
        "experiments/cpu/launch_transformers_accelerate.sh",
        "experiments/cpu/qualification/transformers_accelerate.py",
        "experiments/cpu/qualification/common.py",
    ),
}


def build_frozen_cpu_targets(
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
    for subject, artifact_name in CPU_SUBJECTS.items():
        artifact = root / "artifacts" / "frameworks" / artifact_name
        manifest_path = root / "experiments" / "cpu" / "manifests" / f"{subject}.json"
        manifest = InterventionExecutionManifest.from_path(manifest_path)
        baseline = json.loads(manifest.baseline_config.read_text(encoding="utf-8"))
        graph_payload = json.loads(artifact.read_text(encoding="utf-8"))
        queue = select_interventions(
            DependencyGraph.from_dict(graph_payload),
            baseline,
            limit=limit_per_subject,
            solver_timeout_ms=solver_timeout_ms,
        )
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
            for rank, candidate in enumerate(queue.candidates, start=1)
        ]
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
                    {"path": path, "sha256": _sha256_file(root / path)}
                    for path in CPU_HARNESS_FILES[subject]
                ],
                "selection_summary": dict(queue.to_dict()["summary"]),
                "targets": targets,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 2,
        "name": "cpu-cross-framework-validation-targets",
        "metadata": {
            "selection_basis": "static graph plus qualified CPU baseline only",
            "limit_per_subject": limit_per_subject,
            "solver_timeout_ms": solver_timeout_ms,
            "subject_count": len(subjects),
            "target_count": target_count,
            "scope_note": (
                "Megatron-Core is excluded because the evaluated version does not expose "
                "a CPU training path equivalent to the accelerator runtime used by RQ1."
            ),
        },
        "subjects": subjects,
    }
    payload["frozen"] = {
        "sha256": _payload_sha256(payload),
        "target_count": target_count,
    }
    return payload


def load_frozen_cpu_targets(path: Path) -> dict[str, Any]:
    payload = load_frozen_gpu_targets(path)
    if payload.get("name") != "cpu-cross-framework-validation-targets":
        raise ValueError("frozen target file is not a CPU campaign")
    return payload


def run_frozen_cpu_subject(
    root: Path, frozen: Mapping[str, Any], subject: str
) -> dict[str, Any]:
    result = run_frozen_gpu_subject(root, frozen, subject)
    result["platform"] = "cpu"
    return result


def summarize_frozen_cpu_results(
    frozen: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    *,
    hardware: Mapping[str, Any],
    campaign_date: str,
    runner_revision: str,
    result_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    summary = summarize_frozen_gpu_results(
        frozen,
        results,
        hardware=hardware,
        campaign_date=campaign_date,
        runner_revision=runner_revision,
        result_hashes=result_hashes,
    )
    summary["name"] = "cpu-cross-framework-frozen-validation-summary"
    summary["protocol"]["runtime_scope"] = (
        "real CPU training paths for PyTorch native, DeepSpeed, and Transformers/Accelerate"
    )
    summary["protocol"]["megatron_core_scope"] = (
        "excluded: no equivalent CPU training runtime in the evaluated version"
    )
    return summary


def dump_frozen_cpu_targets(payload: Mapping[str, Any], path: Path) -> None:
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
