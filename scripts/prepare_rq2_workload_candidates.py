#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.workload_candidates import (
    materialize_workload_candidates,
    validate_workload_candidate_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize pinned RQ2 workload candidate baselines from lm-sv assets."
    )
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    args = parser.parse_args()

    result = materialize_workload_candidates(
        args.source_spec,
        args.source_root,
        args.output_dir,
        args.registry_output,
    )
    validation = validate_workload_candidate_manifest(
        result["manifest_path"], result["registry_path"]
    )
    print(
        json.dumps(
            {
                **validation,
                "manifest": result["manifest_path"],
                "registry": result["registry_path"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
