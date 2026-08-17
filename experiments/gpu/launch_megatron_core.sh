#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG_JSON" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ROOT=$(dirname "$REPO_ROOT")
PYTHON=${CONFIGFUZZ_MEGATRON_PYTHON:-"$GPU_ROOT/megatron-env/bin/python"}
MEGATRON_ROOT=${CONFIGFUZZ_MEGATRON_ROOT:-"$GPU_ROOT/Megatron-LM"}
GPUS=${CONFIGFUZZ_GPU_DEVICES:-4,6}
MASTER_PORT=${CONFIGFUZZ_MASTER_PORT:-29656}
NPROC=${CONFIGFUZZ_NPROC_PER_NODE:-2}

export PYTHONPATH="$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
SITE_PACKAGES=$("$PYTHON" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)
if [[ -n "$SITE_PACKAGES" && -d "$SITE_PACKAGES/nvidia" ]]; then
  NVIDIA_LIB_PATH=$(find "$SITE_PACKAGES/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: -)
  if [[ -n "$NVIDIA_LIB_PATH" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTHONUNBUFFERED=1

exec "$PYTHON" -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node="$NPROC" \
  --master-addr=127.0.0.1 \
  --master-port="$MASTER_PORT" \
  "$REPO_ROOT/experiments/gpu/qualification/megatron_core.py" \
  --config "$1"
