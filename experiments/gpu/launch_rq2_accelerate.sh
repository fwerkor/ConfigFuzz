#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 CONFIG_JSON" >&2; exit 2; fi
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ROOT=$(dirname "$REPO_ROOT")
PYTHON=${CONFIGFUZZ_GPU_PYTHON:-"$GPU_ROOT/env/bin/python"}
GPUS=${CONFIGFUZZ_GPU_DEVICES:-4,5}
MASTER_PORT=${CONFIGFUZZ_MASTER_PORT:-29803}
NPROC=${CONFIGFUZZ_NPROC_PER_NODE:-2}
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$GPU_ROOT/transformers/src:$GPU_ROOT/accelerate/src:$REPO_ROOT/experiments/gpu${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
MIXED_PRECISION=$(
  "$PYTHON" - "$1" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
precision = p.get("precision", {})
bf16 = bool(precision.get("bf16", False))
fp16 = bool(precision.get("fp16", False))
print("bf16" if bf16 else "fp16" if fp16 else "no")
PY
)
EXTRA=()
[[ ${CONFIGFUZZ_RQ2_SKIP_CHECKPOINT:-0} == 1 ]] && EXTRA+=(--skip-checkpoint)
exec "$PYTHON" -m accelerate.commands.launch --multi_gpu --num_processes="$NPROC" --main_process_port="$MASTER_PORT" --mixed_precision="$MIXED_PRECISION" "$REPO_ROOT/experiments/gpu/rq2_family_runner.py" --framework accelerate --config "$1" "${EXTRA[@]}"
