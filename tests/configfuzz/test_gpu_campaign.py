from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from configfuzz.gpu_campaign import (
    GPU_SUBJECTS,
    build_frozen_gpu_targets,
    dump_frozen_gpu_targets,
    load_frozen_gpu_targets,
)


ROOT = Path(__file__).resolve().parents[2]


def test_gpu_targets_freeze_deterministically(tmp_path: Path) -> None:
    first = build_frozen_gpu_targets(ROOT, limit_per_subject=2, solver_timeout_ms=2000)
    second = build_frozen_gpu_targets(ROOT, limit_per_subject=2, solver_timeout_ms=2000)

    assert first["frozen"] == second["frozen"]
    assert first["metadata"]["subject_count"] == len(GPU_SUBJECTS)
    assert first["metadata"]["target_count"] == sum(
        len(subject["targets"]) for subject in first["subjects"]
    )
    assert all(subject["targets"] for subject in first["subjects"])

    output = tmp_path / "targets.yaml"
    dump_frozen_gpu_targets(first, output)
    loaded = load_frozen_gpu_targets(output)
    assert loaded["frozen"] == first["frozen"]


def test_gpu_target_hash_detects_tampering(tmp_path: Path) -> None:
    payload = build_frozen_gpu_targets(ROOT, limit_per_subject=1, solver_timeout_ms=2000)
    tampered = deepcopy(payload)
    tampered["subjects"][0]["targets"][0]["expression"] += " and false"
    output = tmp_path / "tampered.yaml"
    output.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_gpu_targets(output)
