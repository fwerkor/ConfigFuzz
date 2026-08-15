#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from configfuzz.gpu_campaign import load_frozen_gpu_targets, summarize_frozen_gpu_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a completed frozen GPU validation campaign.")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--campaign-date", required=True)
    parser.add_argument("--runner-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frozen = load_frozen_gpu_targets(args.targets)
    result_payloads: dict[str, dict[str, object]] = {}
    for item in frozen["subjects"]:
        subject = str(item["subject"])
        path = args.results_dir / f"{subject}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"GPU result root must be an object: {path}")
        result_payloads[subject] = payload

    qualification = yaml.safe_load(args.qualification.read_text(encoding="utf-8"))
    if not isinstance(qualification, dict) or not isinstance(qualification.get("hardware"), dict):
        raise ValueError("qualification file is missing hardware metadata")
    summary = summarize_frozen_gpu_results(
        frozen,
        result_payloads,
        hardware=qualification["hardware"],
        campaign_date=args.campaign_date,
        runner_revision=args.runner_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
