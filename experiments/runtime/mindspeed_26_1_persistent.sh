#!/usr/bin/env bash
set -euo pipefail

PROFILE=${PROFILE:-qwen2}
CANN_ROOT=${CANN_ROOT:-/model/cyh/configfuzz-cann85/cann-8.5.1}
PY_ENV=${PY_ENV:-/model/cyh/configfuzz-llm261-venv}
NNAL_ENV=${NNAL_ENV:-/model/cyh/configfuzz-cann85/nnal/atb/set_env.sh}
MSLLM_ROOT=${MSLLM_ROOT:-/model/cyh/configfuzz-stack/MindSpeed-LLM}
MINDSPEED_ROOT=${MINDSPEED_ROOT:-/model/cyh/configfuzz-stack/MindSpeed}
MEGATRON_ROOT=${MEGATRON_ROOT:-/model/cyh/configfuzz-stack/Megatron-LM}
CONFIGFUZZ_ROOT=${CONFIGFUZZ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
WORKER=${WORKER:-$CONFIGFUZZ_ROOT/experiments/runtime/mindspeed_persistent_worker_26_1.py}
MANIFEST=${MANIFEST:?MANIFEST is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
CASE_LOG_ROOT=${CASE_LOG_ROOT:?CASE_LOG_ROOT is required}
MASTER_PORT=${MASTER_PORT:-6110}
NPROC=${NPROC:-1}
TP=${TP:-1}
PP=${PP:-1}
ALLOW_MULTI_RANK_EXPERIMENTAL=${ALLOW_MULTI_RANK_EXPERIMENTAL:-0}

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
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

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
  --eval-iters 0
  --eval-interval 100
  --ckpt-format torch
  --distributed-backend nccl
  --transformer-impl local
)

PROFILE_ARGS=()
case "$PROFILE" in
  qwen2)
    COMMON_ARGS+=(--vocab-size 2048)
    PROFILE_ARGS+=(
      --group-query-attention
      --num-query-groups 2
      --use-flash-attn
    )
    ;;
  llama2)
    COMMON_ARGS+=(--vocab-size 4096)
    PROFILE_ARGS+=(--use-flash-attn)
    ;;
  chatglm3)
    COMMON_ARGS+=(--vocab-size 4096)
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
    COMMON_ARGS+=(--vocab-size 4096)
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
    COMMON_ARGS+=(--vocab-size 4096)
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

CONTROL_ARGS=(
  --manifest "$MANIFEST"
  --output "$OUTPUT"
  --case-log-root "$CASE_LOG_ROOT"
)
if [ "$ALLOW_MULTI_RANK_EXPERIMENTAL" = "1" ]; then
  CONTROL_ARGS+=(--allow-multi-rank-experimental)
fi

cd "$MSLLM_ROOT"
exec "$PY_ENV/bin/python" -m torch.distributed.run \
  --nproc_per_node "$NPROC" \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  "$WORKER" \
  "${CONTROL_ARGS[@]}" \
  -- \
  "${COMMON_ARGS[@]}" \
  "${PROFILE_ARGS[@]}"
