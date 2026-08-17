#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_ROOT=$(dirname "$REPO_ROOT")
PYTHON=${CONFIGFUZZ_MEGATRON_PYTHON:-"$GPU_ROOT/megatron-env/bin/python"}

SITE_PACKAGES=$(
  "$PYTHON" - <<'PY'
import site
paths = site.getsitepackages()
if not paths:
    raise SystemExit("unable to locate site-packages")
print(paths[0])
PY
)
CUDNN_ROOT="$SITE_PACKAGES/nvidia/cudnn"
NCCL_ROOT="$SITE_PACKAGES/nvidia/nccl"
NVIDIA_LIB_PATH=$(
  find "$SITE_PACKAGES/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null \
    | paste -sd: -
)

for path in "$CUDNN_ROOT/include/cudnn.h" "$NCCL_ROOT/include/nccl.h"; do
  if [[ ! -f "$path" ]]; then
    echo "missing CUDA dependency header: $path" >&2
    echo "install the PyTorch CUDA dependencies in the Megatron validation environment first" >&2
    exit 2
  fi
done

CUDA_HOME=${CUDA_HOME:-$(
  "$PYTHON" - <<'PY'
from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME or "")
PY
)}
if [[ -z "$CUDA_HOME" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "CUDA toolkit with nvcc is required to build transformer-engine-torch" >&2
  exit 2
fi

"$PYTHON" -m pip install ninja
export PATH="$(dirname "$PYTHON"):$PATH"
export MAX_JOBS=${MAX_JOBS:-8}
export CPATH="$CUDNN_ROOT/include:$NCCL_ROOT/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NVIDIA_LIB_PATH${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NVIDIA_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME

"$PYTHON" -m pip install --no-build-isolation \
  -r "$REPO_ROOT/experiments/gpu/requirements-megatron-validation.txt"

"$PYTHON" - <<'PY'
import transformer_engine
import transformer_engine.pytorch
print(f"Transformer Engine {transformer_engine.__version__} is available")
PY
