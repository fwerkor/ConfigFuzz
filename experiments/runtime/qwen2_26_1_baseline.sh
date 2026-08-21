#!/usr/bin/env bash
set -euo pipefail

CANN_ROOT=${CANN_ROOT:-/model/cyh/configfuzz-cann85/cann-8.5.1}
PY_ENV=${PY_ENV:-/model/cyh/configfuzz-llm261-venv}
NNAL_ENV=${NNAL_ENV:-/model/cyh/configfuzz-cann85/nnal/atb/set_env.sh}
MSLLM_ROOT=${MSLLM_ROOT:-/model/cyh/configfuzz-stack/MindSpeed-LLM}
MINDSPEED_ROOT=${MINDSPEED_ROOT:-/model/cyh/configfuzz-stack/MindSpeed}
MEGATRON_ROOT=${MEGATRON_ROOT:-/model/cyh/configfuzz-stack/Megatron-LM}
OUT_ROOT=${OUT_ROOT:-/model/cyh/configfuzz-runs/qwen2-dual-npu-26.1}
MASTER_PORT=${MASTER_PORT:-6011}
NPROC=${NPROC:-2}
TP=${TP:-1}
PP=${PP:-1}
INITIAL_TRAIN_ITERS=${INITIAL_TRAIN_ITERS:-2}
CONTINUE_TRAIN_ITERS=${CONTINUE_TRAIN_ITERS:-3}
SKIP_RELOAD=${SKIP_RELOAD:-0}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}

set +e
set +u
source "$CANN_ROOT/set_env.sh"
CANN_SOURCE_RC=$?
set -u
set -e
if [ "$CANN_SOURCE_RC" -ne 0 ]; then
  exit "$CANN_SOURCE_RC"
fi
set +e
set +u
source "$NNAL_ENV"
NNAL_SOURCE_RC=$?
set -u
set -e
if [ "$NNAL_SOURCE_RC" -ne 0 ]; then
  exit "$NNAL_SOURCE_RC"
fi
export PATH="$PY_ENV/bin:$PATH"
export PYTHONPATH="$MINDSPEED_ROOT:$MEGATRON_ROOT:$MSLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=600
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}

mkdir -p "$OUT_ROOT/checkpoints" "$OUT_ROOT/logs"
cd "$MSLLM_ROOT"

COMMON_ARGS=(
  --use-mcore-models
  --mock-data
  --tokenizer-type NullTokenizer
  --vocab-size 2048
  --make-vocab-size-divisible-by 1
  --tensor-model-parallel-size "$TP"
  --pipeline-model-parallel-size "$PP"
  --num-layers 4
  --hidden-size 512
  --ffn-hidden-size 1024
  --num-attention-heads 8
  --group-query-attention
  --num-query-groups 2
  --seq-length 128
  --max-position-embeddings 128
  --micro-batch-size 1
  --global-batch-size 4
  --position-embedding-type rope
  --normalization RMSNorm
  --swiglu
  --disable-bias-linear
  --use-flash-attn
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
  --eval-iters 0
  --eval-interval 100
  --ckpt-format torch
  --distributed-backend nccl
  --transformer-impl local
)

if [ "$TP" -gt 1 ]; then
  COMMON_ARGS+=(--sequence-parallel)
fi
if [ "$SAVE_CHECKPOINTS" = "1" ]; then
  COMMON_ARGS+=(--save "$OUT_ROOT/checkpoints" --save-interval 1)
fi

run_phase() {
  local phase=$1
  local train_iters=$2
  shift 2
  echo "[ConfigFuzz baseline] phase=$phase train_iters=$train_iters start=$(date --iso-8601=seconds)" | tee -a "$OUT_ROOT/logs/${phase}.log"
  "$PY_ENV/bin/python" -m torch.distributed.run \
    --nproc_per_node "$NPROC" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    pretrain_gpt.py \
    "${COMMON_ARGS[@]}" \
    --train-iters "$train_iters" \
    "$@" 2>&1 | tee -a "$OUT_ROOT/logs/${phase}.log"
}

if [ "${REUSE_EXISTING:-0}" != "1" ]; then
  rm -rf "$OUT_ROOT/checkpoints"
  mkdir -p "$OUT_ROOT/checkpoints"
  run_phase train_and_save "$INITIAL_TRAIN_ITERS" "$@"
fi
if [ "$SKIP_RELOAD" != "1" ]; then
  run_phase reload_and_continue "$CONTINUE_TRAIN_ITERS" --load "$OUT_ROOT/checkpoints" --override-opt_param-scheduler "$@"
fi

echo "[ConfigFuzz baseline] completed=$(date --iso-8601=seconds)" | tee "$OUT_ROOT/COMPLETED"
