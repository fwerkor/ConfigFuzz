#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PYTHON=${CONFIGFUZZ_DRIVER_PYTHON:-"$(dirname "$REPO_ROOT")/env/bin/python"}
GPUS=${CONFIGFUZZ_GPU_DEVICES:-4,5}
DEVICE_COUNT=${CONFIGFUZZ_NPROC_PER_NODE:-2}
SEED=${CONFIGFUZZ_RQ2_SEED:-2026}
OUT=${CONFIGFUZZ_RQ2_OUTPUT_ROOT:-"$REPO_ROOT/formal-results/rq2-gpu-all-qualified-20260821"}
PLAN_ROOT="$OUT/plans"
RUNTIME_ROOT="$OUT/runtime"
ASSEMBLY_ROOT="$OUT/assembly"
mkdir -p "$PLAN_ROOT" "$RUNTIME_ROOT" "$ASSEMBLY_ROOT"

INTENTS=${CONFIGFUZZ_RQ2_INTENTS:-experiments/rq2/intents.qualified.all.frozen.yaml}
OLD_RESULTS=${CONFIGFUZZ_RQ2_REUSE_SOURCE_ROOT:-experiments/rq2/results/gpu-primary-20260819}

plan_subject() {
  local framework=$1
  shift
  "$PYTHON" scripts/plan_rq2_campaign.py \
    --workloads "experiments/rq2/promoted/gpu/$framework/workloads.yaml" \
    --intents "$INTENTS" \
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
  shift 4

  local args=(
    --framework "$framework"
    --plan "$PLAN_ROOT/$framework.json"
    --workloads "experiments/rq2/promoted/gpu/$framework/workloads.yaml"
    --launcher "$launcher"
    --output "$OUT/$framework.jsonl"
    --runtime-root "$RUNTIME_ROOT/$framework"
    --assembly-root "$ASSEMBLY_ROOT/$framework"
    --accelerator-kind gpu
    --devices "$GPUS"
    --device-count "$DEVICE_COUNT"
    --seed "$SEED"
    --master-port "$port"
    --timeout-seconds "$timeout"
    --max-infra-retries 2
    --harness-path "$launcher"
  )
  while (($#)); do
    args+=(--harness-path "$1")
    shift
  done

  local previous="$OLD_RESULTS/$framework.jsonl.gz"
  if [[ -f "$previous" ]]; then
    args+=(--source-result "$previous")
  fi

  "$PYTHON" scripts/run_rq2_reuse_campaign.py "${args[@]}"
}

COMMON_HARNESS=(
  experiments/gpu/rq2_family_runner.py
  experiments/gpu/rq2_family_factory.py
  experiments/gpu/runtime_events.py
)

run_subject pytorch-cuda experiments/gpu/launch_rq2_pytorch.sh 90 30300 "${COMMON_HARNESS[@]}"
run_subject deepspeed experiments/gpu/launch_rq2_deepspeed.sh 120 30500 "${COMMON_HARNESS[@]}"
run_subject transformers-accelerate experiments/gpu/launch_rq2_accelerate.sh 120 30700 "${COMMON_HARNESS[@]}"
run_subject megatron-core experiments/gpu/launch_rq2_megatron.sh 90 30900 \
  experiments/gpu/rq2_megatron_runner.py \
  experiments/gpu/runtime_events.py
