#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 {pytorch-native|deepspeed|transformers-accelerate|megatron-core} [ROUNDS]" >&2
  exit 2
fi

SUBJECT=$1
ROUNDS=${2:-10}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${CONFIGFUZZ_DRIVER_PYTHON:-python}

case "$SUBJECT" in
  pytorch-native)
    GRAPH=artifacts/frameworks/pytorch_v2.13.0.json
    ;;
  deepspeed)
    GRAPH=artifacts/frameworks/deepspeed_v0.19.1.json
    ;;
  transformers-accelerate)
    GRAPH=artifacts/frameworks/transformers_v5.9.0_accelerate_v1.14.0.json
    ;;
  megatron-core)
    GRAPH=artifacts/frameworks/megatron_core_v0.18.2.json
    ;;
  *)
    echo "unknown GPU subject: $SUBJECT" >&2
    exit 2
    ;;
esac

mkdir -p "$REPO_ROOT/artifacts/gpu/results"
cd "$REPO_ROOT"
exec "$PYTHON" -m configfuzz active-validate \
  "$GRAPH" \
  "experiments/gpu/manifests/$SUBJECT.json" \
  --rounds "$ROUNDS" \
  --output "artifacts/gpu/results/$SUBJECT-active-validation.json"
