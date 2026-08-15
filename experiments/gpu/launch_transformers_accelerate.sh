#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG_JSON" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ROOT=$(dirname "$REPO_ROOT")
PYTHON=${CONFIGFUZZ_GPU_PYTHON:-"$GPU_ROOT/env/bin/python"}
TRANSFORMERS_ROOT=${CONFIGFUZZ_TRANSFORMERS_ROOT:-"$GPU_ROOT/transformers"}
ACCELERATE_ROOT=${CONFIGFUZZ_ACCELERATE_ROOT:-"$GPU_ROOT/accelerate"}
GPUS=${CONFIGFUZZ_GPU_DEVICES:-4,6}
MASTER_PORT=${CONFIGFUZZ_MASTER_PORT:-29655}
NPROC=${CONFIGFUZZ_NPROC_PER_NODE:-2}
MIXED_PRECISION=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["mixed_precision"])' "$1"
)

export PYTHONPATH="$TRANSFORMERS_ROOT/src:$ACCELERATE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

exec "$PYTHON" -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes="$NPROC" \
  --main_process_port="$MASTER_PORT" \
  --mixed_precision="$MIXED_PRECISION" \
  "$REPO_ROOT/experiments/gpu/qualification/transformers_accelerate.py" \
  --config "$1"
