#!/usr/bin/env bash
set -euo pipefail

PROFILE=${PROFILE:-llama2}
CANN_ROOT=${CANN_ROOT:-/model/cyh/configfuzz-cann85/cann-8.5.1}
NNAL_ENV=${NNAL_ENV:-/model/cyh/configfuzz-cann85/nnal/atb/set_env.sh}
PY_ENV=${PY_ENV:-/model/cyh/configfuzz-llm261-venv}
MSLLM_ROOT=${MSLLM_ROOT:-/model/cyh/configfuzz-stack/MindSpeed-LLM}
MINDSPEED_ROOT=${MINDSPEED_ROOT:-/model/cyh/configfuzz-stack/MindSpeed}
MEGATRON_ROOT=${MEGATRON_ROOT:-/model/cyh/configfuzz-stack/Megatron-LM}
OUT_ROOT=${OUT_ROOT:-/model/cyh/configfuzz-runs/${PROFILE}-dual-npu-26.1}
MASTER_PORT=${MASTER_PORT:-6020}
NPROC=${NPROC:-2}
TP=${TP:-1}
PP=${PP:-1}
INITIAL_TRAIN_ITERS=${INITIAL_TRAIN_ITERS:-2}
CONTINUE_TRAIN_ITERS=${CONTINUE_TRAIN_ITERS:-3}
SKIP_RELOAD=${SKIP_RELOAD:-0}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}

safe_source() {
  local f=$1
  set +e
  set +u
  # shellcheck disable=SC1090
  source "$f"
  local rc=$?
  set -u
  set -e
  return "$rc"
}

safe_source "$CANN_ROOT/set_env.sh"
safe_source "$NNAL_ENV"
export PATH="$PY_ENV/bin:$PATH"
export PYTHONPATH="$MINDSPEED_ROOT:$MEGATRON_ROOT:$MSLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}

mkdir -p "$OUT_ROOT/checkpoints" "$OUT_ROOT/logs"
cd "$MSLLM_ROOT"

COMMON_ARGS=(
  --use-mcore-models
  --mock-data
  --tokenizer-type NullTokenizer
  --vocab-size 4096
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
  --eval-iters 0
  --eval-interval 100
  --ckpt-format torch
  --distributed-backend nccl
  --transformer-impl local
)

PROFILE_ARGS=()
case "$PROFILE" in
  llama2)
    PROFILE_ARGS+=(--use-flash-attn)
    ;;
  chatglm3)
    PROFILE_ARGS+=(
      --group-query-attention
      --num-query-groups 2
      --add-qkv-bias
      --use-glm-rope
      --rotary-percent 0.5
      --use-flash-attn
    )
    ;;
  mixtral)
    PROFILE_ARGS+=(
      --group-query-attention
      --num-query-groups 2
      --num-experts 4
      --moe-router-topk 2
      --moe-ffn-hidden-size 512
      --moe-router-load-balancing-type aux_loss
      --moe-aux-loss-coeff 0.01
      --moe-grouped-gemm
      --moe-token-dispatcher-type allgather
      --expert-model-parallel-size 1
      --use-flash-attn
    )
    ;;
  deepseekv3)
    PROFILE_ARGS+=(
      --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec
      --multi-latent-attention
      --qk-pos-emb-head-dim 64
      --qk-head-dim 64
      --q-lora-rank 128
      --kv-lora-rank 64
      --v-head-dim 64
      --qk-layernorm
      --num-experts 4
      --moe-router-topk 2
      --moe-ffn-hidden-size 256
      --moe-router-load-balancing-type aux_loss
      --moe-aux-loss-coeff 0.01
      --moe-grouped-gemm
      --moe-token-dispatcher-type allgather
      --expert-model-parallel-size 1
      --first-k-dense-replace 1
      --moe-layer-freq 1
      --n-shared-experts 1
      --untie-embeddings-and-output-weights
      --attention-softmax-in-fp32
      --no-masked-softmax-fusion
      --use-flash-attn
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
if [ "$SAVE_CHECKPOINTS" = "1" ]; then
  COMMON_ARGS+=(--save "$OUT_ROOT/checkpoints" --save-interval 1)
fi

run_phase() {
  local phase=$1
  local train_iters=$2
  shift 2
  echo "[ConfigFuzz baseline] profile=$PROFILE phase=$phase train_iters=$train_iters start=$(date --iso-8601=seconds)" | tee -a "$OUT_ROOT/logs/${phase}.log"
  "$PY_ENV/bin/python" -m torch.distributed.run \
    --nproc_per_node "$NPROC" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    pretrain_gpt.py \
    "${COMMON_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
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

echo "[ConfigFuzz baseline] profile=$PROFILE completed=$(date --iso-8601=seconds)" | tee "$OUT_ROOT/COMPLETED"
