#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_FUSION_OP_ENABLE=0
# NPU 算子编译缓存目录，不设则默认为当前目录下的 kernel_meta/，设到独立目录便于清理且不污染项目
export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-$(cd -- "$(dirname "$0")" && pwd)/cache_ascend}"

NPUS_PER_NODE=1
MASTER_ADDR=localhost
# 避免上次崩溃后端口仍被占用导致 Bind_Failed；随机 27000–27999 或可改为固定如 27001
MASTER_PORT=${MASTER_PORT:-$((27000 + RANDOM % 1000))}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
CP=1
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


GPT_ARGS="
    --num-layers 1 \
    --hidden-size 32 \
    --num-attention-heads 16 \
    --num-query-groups 16 \
    --ffn-hidden-size 64 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --tokenizer-type NullTokenizer \
    --vocab-size 32000 \
    --position-embedding-type rope \
    --no-masked-softmax-fusion \
    --lr 0.001 \
    --train-iters 10 \
    --lr-decay-style cosine \
    --weight-decay 0.0 \
    --lr-warmup-fraction 0.03 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --bf16 \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 1 \
"

# 结果基目录：默认 ./results/_module_combine，可用环境变量 RESULTS_DIR 或命令行 --results-dir 指定
# 默认每次在基目录下创建 mutate_YYYYMMDD_HHMMSS；若指定 --dir-name，则使用基目录/dir-name（train.log 与 mutate.py 一致）
RESULTS_BASE=${RESULTS_DIR:-./results/_module_combine}
run_name=$(date +%Y%m%d_%H%M%S)

# 透传给 mutate.py 的 extra args（会去掉 --results-dir / --dir-name，由下方统一传入）
# 用法：./mm_mutate.sh [mutate.py args...]
# 例：  ./mm_mutate.sh --rounds 3
# 例：  ./mm_mutate.sh --results-dir ./results/my_run --rounds 2
# 例：  ./mm_mutate.sh --results-dir ./out --dir-name my_exp --rounds 2
EXTRA_MUTATE_ARGS=("$@")

USER_RESULTS_DIR=""
USER_DIR_NAME=""
for ((i=0; i<${#EXTRA_MUTATE_ARGS[@]}; i++)); do
  arg="${EXTRA_MUTATE_ARGS[$i]}"
  if [[ "$arg" == "--results-dir" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_MUTATE_ARGS[@]} )); then
      USER_RESULTS_DIR="${EXTRA_MUTATE_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --results-dir=* ]]; then
    USER_RESULTS_DIR="${arg#--results-dir=}"
  elif [[ "$arg" == "--dir-name" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_MUTATE_ARGS[@]} )); then
      USER_DIR_NAME="${EXTRA_MUTATE_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --dir-name=* ]]; then
    USER_DIR_NAME="${arg#--dir-name=}"
  fi
done

if [[ -n "$USER_RESULTS_DIR" ]]; then
  RESULTS_BASE="$USER_RESULTS_DIR"
fi
if [[ -n "$USER_DIR_NAME" ]]; then
  RUN_DIR="${RESULTS_BASE}/${USER_DIR_NAME}"
else
  RUN_DIR="${RESULTS_BASE}/mutate_${run_name}"
fi

# 去掉 EXTRA_MUTATE_ARGS 中的脚本专用参数，避免与解析后的 RUN_DIR 冲突
FILTERED_MUTATE_ARGS=()
skip_next=0
for ((i=0; i<${#EXTRA_MUTATE_ARGS[@]}; i++)); do
  if (( skip_next )); then
    skip_next=0
    continue
  fi
  arg="${EXTRA_MUTATE_ARGS[$i]}"
  if [[ "$arg" == "--results-dir" ]] || [[ "$arg" == "--dir-name" ]]; then
    skip_next=1
    continue
  fi
  if [[ "$arg" == --results-dir=* ]] || [[ "$arg" == --dir-name=* ]]; then
    continue
  fi
  FILTERED_MUTATE_ARGS+=("$arg")
done

mkdir -p "${RUN_DIR}"

# 清理可能遗留的 torch_extensions 锁文件，避免因上次崩溃导致本次一直卡在 "Using ... as PyTorch extensions root"
TORCH_EXT_CACHE="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}"
for lock in "${TORCH_EXT_CACHE}"/py310_cpu/*/lock; do
    [ -f "$lock" ] && rm -f "$lock" && echo "[env] removed stale lock: $lock"
done 2>/dev/null

# 自定义目录名时路径已是最终 run 目录，mutate.py 不应再套一层 mutate_*
NO_SUBDIR_ARG=()
if [[ -n "$USER_DIR_NAME" ]]; then
  NO_SUBDIR_ARG=(--no-create-results-subdir)
fi

export LMSV_MM_MSARUN=0
export LMSV_MM_PTARUN=1

torchrun $DISTRIBUTED_ARGS \
    mutate.py \
    $GPT_ARGS \
    --results-dir "${RUN_DIR}" \
    "${NO_SUBDIR_ARG[@]}" \
    --distributed-backend nccl \
    "${FILTERED_MUTATE_ARGS[@]}" \
    2>&1 | tee "${RUN_DIR}/train.log"
chmod 440 "${RUN_DIR}/train.log"
