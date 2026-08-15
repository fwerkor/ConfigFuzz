#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG_JSON" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ROOT=$(dirname "$REPO_ROOT")
PYTHON=${CONFIGFUZZ_GPU_PYTHON:-"$GPU_ROOT/env/bin/python"}
GPUS=${CONFIGFUZZ_GPU_DEVICES:-4,6}
MASTER_PORT=${CONFIGFUZZ_MASTER_PORT:-29653}
NPROC=${CONFIGFUZZ_NPROC_PER_NODE:-2}

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

exec "$PYTHON" -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node="$NPROC" \
  --master-addr=127.0.0.1 \
  --master-port="$MASTER_PORT" \
  "$REPO_ROOT/experiments/gpu/qualification/pytorch_native.py" \
  --config "$1"
