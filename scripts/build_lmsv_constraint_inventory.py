#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import yaml

from configfuzz.extractors import scan_python_paths


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "lmsv_rec" / "mutable_params_pool.yaml"
OUTPUT = ROOT / "artifacts" / "lmsv_static_inventory.json"

CORE_PARAMETERS = {
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "context_parallel_size",
    "sequence_parallel",
    "hidden_size",
    "num_attention_heads",
    "ffn_hidden_size",
    "num_layers",
    "global_batch_size",
    "micro_batch_size",
    "seq_length",
    "max_position_embeddings",
    "vocab_size",
    "num_experts",
    "moe_router_topk",
    "learning_rate",
    "min_lr",
    "lr_warmup_fraction",
    "recompute_method",
    "recompute_granularity",
}


def load_parameters() -> list[str]:
    data = yaml.safe_load(POOL.read_text(encoding="utf-8")) or {}
    parameters = set(CORE_PARAMETERS)
    for section in ("numeric_params", "enum_params"):
        values = data.get(section, {})
        if isinstance(values, dict):
            parameters.update(str(name) for name in values)
    return sorted(parameters)


def main() -> int:
    parameters = load_parameters()
    roots = [
        ROOT / "lmsv_rec" / "utils" / "runtime",
        ROOT / "mm-new" / "net_mutation",
        ROOT / "module_combination_mutation",
    ]
    python_files = sorted(
        path
        for root in roots
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    source_index = {
        path: path.read_text(encoding="utf-8", errors="replace")
        for path in python_files
    }
    results = []
    for parameter in parameters:
        candidate_files = [
            path for path, source in source_index.items() if parameter in source
        ]
        result = scan_python_paths(candidate_files, parameter=parameter)
        if result.constraints:
            results.append(result.to_dict())

    payload = {
        "schema_version": 1,
        "baseline": {
            "repository": "fwerkor/lm-sv",
            "commit": "e73ba3d35",
            "branch": "dev_0.1.0",
        },
        "parameters_considered": len(parameters),
        "parameters_with_candidates": len(results),
        "python_files_indexed": len(python_files),
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"parameters: {len(parameters)}, with candidates: {len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
