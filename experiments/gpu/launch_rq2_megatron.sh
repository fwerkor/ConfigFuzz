#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 CONFIG_JSON" >&2; exit 2; fi
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU_ROOT=$(dirname "$REPO_ROOT")
PYTHON=${CONFIGFUZZ_MEGATRON_PYTHON:-"$GPU_ROOT/megatron-env/bin/python"}
MEGATRON_ROOT=${CONFIGFUZZ_MEGATRON_ROOT:-"$GPU_ROOT/Megatron-LM"}
GPUS=${CONFIGFUZZ_GPU_DEVICES:-4,5}
MASTER_PORT=${CONFIGFUZZ_MASTER_PORT:-29804}
NPROC=${CONFIGFUZZ_NPROC_PER_NODE:-2}
export CUDA_VISIBLE_DEVICES="$GPUS"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTHONPATH="$MEGATRON_ROOT:$REPO_ROOT/experiments/gpu${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
EXTRA=()
[[ ${CONFIGFUZZ_RQ2_SKIP_CHECKPOINT:-0} == 1 ]] && EXTRA+=(--skip-checkpoint)
exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node="$NPROC" --master-addr=127.0.0.1 --master-port="$MASTER_PORT" "$REPO_ROOT/experiments/gpu/rq2_megatron_runner.py" --config "$1" "${EXTRA[@]}"
