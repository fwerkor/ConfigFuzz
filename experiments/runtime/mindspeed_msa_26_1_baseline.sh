#!/usr/bin/env bash
set -euo pipefail

PROFILE=${PROFILE:-qwen2}
CANN_ROOT=${CANN_ROOT:-/model/cyh/configfuzz-cann91/cann-9.1.0}
NNAL_ENV=${NNAL_ENV:-/model/cyh/configfuzz-cann91/nnal/atb/set_env.sh}
PY_ENV=${PY_ENV:-/model/cyh/configfuzz-cann91/envs/msa}
MSA_STACK_ROOT=${MSA_STACK_ROOT:-/model/cyh/configfuzz-stack/msa-run-20260822}
MSADAPTER_ROOT=${MSADAPTER_ROOT:-$MSA_STACK_ROOT/MSAdapter}
MSLLM_ROOT=${MSLLM_ROOT:-$MSA_STACK_ROOT/MindSpeed-LLM}
MINDSPEED_ROOT=${MINDSPEED_ROOT:-$MSA_STACK_ROOT/MindSpeed}
MEGATRON_ROOT=${MEGATRON_ROOT:-$MSA_STACK_ROOT/Megatron-LM}
OUT_ROOT=${OUT_ROOT:-/model/cyh/configfuzz-runs/rq1-msa-${PROFILE}-26.1}
MASTER_PORT=${MASTER_PORT:-6710}
NPROC=${NPROC:-2}
TP=${TP:-1}
PP=${PP:-1}
INITIAL_TRAIN_ITERS=${INITIAL_TRAIN_ITERS:-1}

safe_source() {
  local f=$1
  shift || true
  set +e
  set +u
  # shellcheck disable=SC1090
  source "$f" "$@"
  local rc=$?
  set -u
  set -e
  return "$rc"
}

safe_source "$CANN_ROOT/set_env.sh"
safe_source "$NNAL_ENV" --cxx_abi=0 || safe_source "$NNAL_ENV"
export PATH="$PY_ENV/bin:$PATH"
export PYTHONPATH="$MSADAPTER_ROOT:$MSADAPTER_ROOT/msa_thirdparty:$MSLLM_ROOT:$MINDSPEED_ROOT:$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export MS_ALLOC_CONF=${MS_ALLOC_CONF:-enable_vmm:True}

mkdir -p "$OUT_ROOT"
MSRUN_LOG_DIR="$OUT_ROOT/msrun_log"
rm -rf "$MSRUN_LOG_DIR"
mkdir -p "$MSRUN_LOG_DIR"
cd "$MSLLM_ROOT"

COMMON_ARGS=(
  --use-mcore-models
  --mock-data
  --tokenizer-type NullTokenizer
  --make-vocab-size-divisible-by 1
  --tensor-model-parallel-size "$TP"
  --pipeline-model-parallel-size "$PP"
  --num-layers 4
  --hidden-size 512
  --ffn-hidden-size 1024
  --num-attention-heads 8
  --seq-length 128
  --max-position-embeddings 128
  --micro-batch-size 1
  --global-batch-size 4
  --position-embedding-type rope
  --normalization RMSNorm
  --swiglu
  --disable-bias-linear
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --lr 1e-4
  --min-lr 1e-5
  --lr-decay-style constant
  --weight-decay 0.0
  --clip-grad 1.0
  --bf16
  --seed 42
  --log-interval 1
  --exit-interval 1
  --eval-iters 0
  --eval-interval 100
  --ckpt-format torch
  --distributed-backend nccl
  --transformer-impl local
  --no-gradient-accumulation-fusion
  --no-masked-softmax-fusion
  --no-shared-storage
)

PROFILE_ARGS=()
case "$PROFILE" in
  qwen2)
    COMMON_ARGS+=(--vocab-size 2048)
    PROFILE_ARGS+=(
      --group-query-attention
      --num-query-groups 2
    )
    ;;
  mixtral)
    COMMON_ARGS+=(--vocab-size 4096)
    PROFILE_ARGS+=(
      --group-query-attention
      --num-query-groups 2
      --num-experts 4
      --moe-router-topk 2
      --moe-ffn-hidden-size 512
      --moe-router-load-balancing-type aux_loss
      --moe-aux-loss-coeff 0.01
      --moe-token-dispatcher-type allgather
      --expert-model-parallel-size 1
    )
    ;;
  *)
    echo "unknown PROFILE=$PROFILE" >&2
    exit 2
    ;;
esac

if [ "$TP" -gt 1 ]; then
  COMMON_ARGS+=(--sequence-parallel)
fi

DISTRIBUTED_ARGS=(
  --local_worker_num "$NPROC"
  --worker_num "$NPROC"
  --node_rank 0
  --master_addr 127.0.0.1
  --master_port "$MASTER_PORT"
  --log_dir "$MSRUN_LOG_DIR"
  --join=True
)

echo "[ConfigFuzz MSA baseline] profile=$PROFILE nproc=$NPROC train_iters=$INITIAL_TRAIN_ITERS start=$(date --iso-8601=seconds)"
set +e
msrun "${DISTRIBUTED_ARGS[@]}" pretrain_gpt.py \
  "${COMMON_ARGS[@]}" \
  "${PROFILE_ARGS[@]}" \
  --train-iters "$INITIAL_TRAIN_ITERS" \
  --ai-framework mindspore \
  "$@"
rc=$?
set -e

for log in "$MSRUN_LOG_DIR"/worker_*.log "$MSRUN_LOG_DIR"/*.log; do
  [ -f "$log" ] || continue
  echo "===== $log ====="
  cat "$log"
done

if [ "$rc" -eq 0 ]; then
  echo "[ConfigFuzz MSA baseline] completed=$(date --iso-8601=seconds)" | tee "$OUT_ROOT/COMPLETED"
fi
exit "$rc"
