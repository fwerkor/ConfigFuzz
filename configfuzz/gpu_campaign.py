from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from configfuzz.dependencies import DependencyGraph
from configfuzz.feedback import apply_probe_feedback
from configfuzz.intervention_runner import InterventionExecutionManifest, run_intervention
from configfuzz.selection import select_interventions


GPU_SUBJECTS: dict[str, str] = {
    "pytorch-native": "pytorch_v2.13.0.json",
    "deepspeed": "deepspeed_v0.19.1.json",
    "transformers-accelerate": "transformers_v5.9.0_accelerate_v1.14.0.json",
    "megatron-core": "megatron_core_v0.18.2.json",
}


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
                "selection_summary": queue.to_dict()["summary"],
                "targets": targets,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
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

    graph = DependencyGraph.from_dict(json.loads(artifact.read_text(encoding="utf-8")))
    manifest = InterventionExecutionManifest.from_path(manifest_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rounds: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    for index, target in enumerate(subject_payload.get("targets", ()), start=1):
        if not isinstance(target, Mapping) or not isinstance(target.get("intervention"), Mapping):
            raise ValueError(f"{subject} target {index} is malformed")
        samples = run_intervention({"intervention": target["intervention"]}, manifest)
        outcome_counts.update(sample.outcome.label.value for sample in samples)
        feedback = apply_probe_feedback(graph, baseline, samples)
        rounds.append(
            {
                "index": index,
                "edge_id": str(target["edge_id"]),
                "expression": str(target["expression"]),
                "selection_rank": int(target["rank"]),
                "samples": [sample.to_dict() for sample in samples],
                "feedback": feedback.to_dict(),
            }
        )

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
            "paired_interventions": sum(item["feedback"]["paired_interventions"] for item in rounds),
        },
        "rounds": rounds,
        "dependency_graph": graph.to_dict(),
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
