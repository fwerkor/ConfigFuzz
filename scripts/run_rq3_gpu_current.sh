#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PYTHON=${CONFIGFUZZ_DRIVER_PYTHON:-"$(dirname "$REPO_ROOT")/env/bin/python"}
GPUS=${CONFIGFUZZ_GPU_DEVICES:-4,5}
DEVICE_COUNT=${CONFIGFUZZ_NPROC_PER_NODE:-2}
SEED=${CONFIGFUZZ_RQ3_SEED:-2026}
OUT=${CONFIGFUZZ_RQ3_OUTPUT_ROOT:-"$(dirname "$REPO_ROOT")/rq3-results/current-version-gpu45-20260820"}
PLAN_ROOT="$OUT/plans"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PLAN_ROOT" "$OUT/cases"

plan_subject() {
  local framework=$1
  shift
  local plan="$PLAN_ROOT/$framework.json"
  if [[ -f "$plan" ]]; then
    return
  fi
  "$PYTHON" scripts/plan_rq2_campaign.py \
    --workloads "experiments/rq2/promoted/gpu/$framework/workloads.yaml" \
    --intents experiments/rq2/intents.prequalified.frozen.yaml \
    --output "$plan" \
    --world-size "$DEVICE_COUNT" \
    "$@"
}

run_subject() {
  local framework=$1
  local launcher=$2
  local timeout=$3
  local port=$4
  echo "RQ3_CURRENT_START framework=$framework gpus=$GPUS" >&2
  "$PYTHON" scripts/run_rq3_gpu_current.py \
    --framework "$framework" \
    --plan "$PLAN_ROOT/$framework.json" \
    --workloads "experiments/rq2/promoted/gpu/$framework/workloads.yaml" \
    --launcher "$launcher" \
    --output-root "$OUT/cases/$framework" \
    --output "$OUT/$framework.jsonl" \
    --gpus "$GPUS" \
    --device-count "$DEVICE_COUNT" \
    --seed "$SEED" \
    --master-port "$port" \
    --timeout-seconds "$timeout" \
    --max-generated-tests-per-method 2000 \
    --max-accelerator-hours-per-method 24
}

# Keep the same frozen intent universe and method ordering used by RQ2.  The
# RQ3 runner differs by executing checkpoint save/load and by applying the RQ3
# per-framework/per-method search budgets.
plan_subject deepspeed
plan_subject transformers-accelerate
plan_subject megatron-core \
  --workload qwen2-train \
  --workload llama2-train \
  --workload mixtral-train
plan_subject pytorch-cuda

# Start with framework stacks where current-version configuration defects have
# already been observed, then cover the remaining GPU subject.
run_subject deepspeed experiments/gpu/launch_rq2_deepspeed.sh 180 32300
run_subject transformers-accelerate experiments/gpu/launch_rq2_accelerate.sh 180 32600
run_subject megatron-core experiments/gpu/launch_rq2_megatron.sh 150 32900
run_subject pytorch-cuda experiments/gpu/launch_rq2_pytorch.sh 150 33200
