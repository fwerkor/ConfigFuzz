#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 CONFIG_JSON" >&2; exit 2; }
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${CONFIGFUZZ_CPU_PYTHON:-python}
NPROC=${CONFIGFUZZ_CPU_NPROC:-2}
MASTER_PORT=${CONFIGFUZZ_MASTER_PORT:-30654}
export PYTHONPATH="$REPO_ROOT/.cpu-site${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export DS_ACCELERATOR=cpu
exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node="$NPROC" --master-addr=127.0.0.1 --master-port="$MASTER_PORT" "$REPO_ROOT/experiments/cpu/qualification/deepspeed_runner.py" --config "$1"
