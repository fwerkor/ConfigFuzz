#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_EXPERIMENTS = REPO_ROOT / "experiments" / "gpu"
if str(GPU_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(GPU_EXPERIMENTS))

from rq2_family_factory import build_model, load_profile, model_parameter_count  # noqa: E402
from rq2_family_runner import _forward_loss  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate all canonical RQ2 model families without an accelerator.")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "rq2" / "baselines" / "canonical-v1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = []
    failures = []
    for path in sorted(args.baseline_dir.glob("*.json")):
        try:
            profile = load_profile(path)
            empty_model = build_model(profile, empty_weights=True)
            parameter_count = model_parameter_count(empty_model)

            smoke = copy.deepcopy(profile)
            smoke["precision"] = {"bf16": False, "fp16": False}
            smoke["training"]["micro_batch_size"] = 1
            if smoke["family"] == "cogvideox_video_text":
                smoke["video"]["frames"] = 3
                smoke["video"]["sample_height"] = 8
                smoke["video"]["sample_width"] = 8
                smoke["video"]["max_text_seq_length"] = 8
            else:
                smoke["model"]["seq_length"] = min(24, int(smoke["model"]["seq_length"]))
            torch.manual_seed(2026)
            model = build_model(smoke).float().cpu()
            model.train()
            loss = _forward_loss(smoke, model, torch.device("cpu"), 0, 0)
            loss.backward()
            records.append(
                {
                    "workload_id": profile["workload_id"],
                    "family": profile["family"],
                    "model_class": type(model).__name__,
                    "parameter_count": parameter_count,
                    "status": "model_forward_backward_passed",
                }
            )
        except Exception as exc:  # preflight must report every failed family
            failures.append(
                {
                    "path": str(path),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    payload = {
        "schema_version": 1,
        "name": "rq2-accelerator-free-model-preflight",
        "baseline_dir": _display_path(args.baseline_dir),
        "passed": len(records),
        "failed": len(failures),
        "records": records,
        "failures": failures,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 1 if failures else 0


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
