#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 CONFIG_JSON" >&2; exit 2; }
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${CONFIGFUZZ_CPU_PYTHON:-python}
export PYTHONPATH="$REPO_ROOT/.cpu-site${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "$PYTHON" "$REPO_ROOT/experiments/cpu/qualification/transformers_accelerate.py" --config "$1"
