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
exec "$PYTHON" -m accelerate.commands.launch --multi_gpu --num_processes="$NPROC" --main_process_port="$MASTER_PORT" --mixed_precision=bf16 "$REPO_ROOT/experiments/gpu/rq2_family_runner.py" --framework accelerate --config "$1"
