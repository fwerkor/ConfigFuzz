from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_rq3_gpu_current.py"
SPEC = importlib.util.spec_from_file_location("run_rq3_gpu_current", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cleanup_checkpoint_tree_removes_only_checkpoint_artifacts(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    checkpoint_root = case_root / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "model.bin").write_bytes(b"checkpoint")
    (case_root / "config.json").write_text("{}\n", encoding="utf-8")
    (case_root / "run.log").write_text("done\n", encoding="utf-8")

    assert MODULE._cleanup_checkpoint_tree(case_root) is True
    assert not checkpoint_root.exists()
    assert (case_root / "config.json").is_file()
    assert (case_root / "run.log").is_file()


def test_cleanup_checkpoint_tree_is_idempotent(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    case_root.mkdir()

    assert MODULE._cleanup_checkpoint_tree(case_root) is False
    assert MODULE._cleanup_checkpoint_tree(case_root) is False
