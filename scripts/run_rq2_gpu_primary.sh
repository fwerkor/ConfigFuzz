#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PYTHON=${CONFIGFUZZ_DRIVER_PYTHON:-"$(dirname "$REPO_ROOT")/env/bin/python"}
GPUS=${CONFIGFUZZ_GPU_DEVICES:-4,5}
DEVICE_COUNT=${CONFIGFUZZ_NPROC_PER_NODE:-2}
SEED=${CONFIGFUZZ_RQ2_SEED:-2026}
OUT=${CONFIGFUZZ_RQ2_OUTPUT_ROOT:-"$REPO_ROOT/formal-results/rq2-gpu-primary-20260817"}
PLAN_ROOT="$OUT/plans"
mkdir -p "$PLAN_ROOT" "$OUT/cases"

plan_subject() {
  local framework=$1
  shift
  "$PYTHON" scripts/plan_rq2_campaign.py \
    --workloads "experiments/rq2/promoted/gpu/$framework/workloads.yaml" \
    --intents experiments/rq2/intents.prequalified.frozen.yaml \
    --output "$PLAN_ROOT/$framework.json" \
    --world-size "$DEVICE_COUNT" \
    "$@"
}

plan_subject pytorch-cuda
plan_subject deepspeed
plan_subject transformers-accelerate
plan_subject megatron-core \
  --workload qwen2-train \
  --workload llama2-train \
  --workload mixtral-train

run_subject() {
  local framework=$1
  local launcher=$2
  local timeout=$3
  local port=$4
  "$PYTHON" scripts/run_rq2_gpu_campaign.py \
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
    --timeout-seconds "$timeout"
}

run_subject pytorch-cuda experiments/gpu/launch_rq2_pytorch.sh 90 30300
run_subject deepspeed experiments/gpu/launch_rq2_deepspeed.sh 120 30500
run_subject transformers-accelerate experiments/gpu/launch_rq2_accelerate.sh 120 30700
run_subject megatron-core experiments/gpu/launch_rq2_megatron.sh 90 30900
