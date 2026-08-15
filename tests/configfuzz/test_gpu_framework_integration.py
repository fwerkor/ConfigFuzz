from __future__ import annotations

import json
from pathlib import Path

import pytest

from configfuzz.dependencies import DependencyGraph
from configfuzz.intervention_runner import InterventionExecutionManifest
from configfuzz.selection import select_interventions


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("subject", "artifact"),
    (
        ("pytorch-native", "pytorch_v2.13.0.json"),
        ("deepspeed", "deepspeed_v0.19.1.json"),
        ("transformers-accelerate", "transformers_v5.9.0_accelerate_v1.14.0.json"),
        ("megatron-core", "megatron_core_v0.18.2.json"),
    ),
)
def test_gpu_subject_has_executable_recovered_edge(subject: str, artifact: str) -> None:
    manifest = InterventionExecutionManifest.from_path(
        ROOT / "experiments" / "gpu" / "manifests" / f"{subject}.json"
    )
    graph_payload = json.loads(
        (ROOT / "artifacts" / "frameworks" / artifact).read_text(encoding="utf-8")
    )
    baseline = json.loads(manifest.baseline_config.read_text(encoding="utf-8"))

    queue = select_interventions(
        DependencyGraph.from_dict(graph_payload),
        baseline,
        limit=1,
        solver_timeout_ms=2000,
    )

    assert queue.candidates, f"{subject} has no executable recovered edge"
    assert manifest.cwd == ROOT.resolve()
