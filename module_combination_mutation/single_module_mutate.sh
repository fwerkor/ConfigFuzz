#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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
export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-${SCRIPT_DIR}/cache_ascend}"

NPUS_PER_NODE="${NPUS_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
# 避免上次崩溃后端口仍被占用导致 Bind_Failed；随机 27000–27999 或可改为固定如 27001
MASTER_PORT="${MASTER_PORT:-$((27000 + RANDOM % 1000))}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE="$((${NPUS_PER_NODE} * ${NNODES}))"

TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-1}"
GBS="$((WORLD_SIZE * MBS / CP))"

ALL_MODULES="${ALL_MODULES:-./modules.json}"
ROUNDS="${ROUNDS:-10}"
ITERATIONS="${ITERATIONS:-10}"

# 可选：只对某类模块做变异（传给 single_module_mutate.py 的 --type）
# 允许两种方式：
# 1) 环境变量 TYPE=text_decoder / image_encoder
# 2) 脚本命令行参数 --type text_decoder / image_encoder
#
# 结果目录：默认在 RESULTS_BASE 下创建 mutate_single_YYYYMMDD_HHMMSS；
# 若指定 --dir-name，则使用 RESULTS_BASE/dir-name（与 mm_mutate.sh 一致）。
# 可选：--results-dir 覆盖 RESULTS_BASE（环境变量 RESULTS_DIR 仍为默认基目录）。
TYPE="${TYPE:-}"
USER_RESULTS_DIR=""
USER_DIR_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)
            if [[ $# -lt 2 ]]; then
                echo "[error] --type requires a value (e.g. --type text_decoder)" >&2
                exit 1
            fi
            TYPE="$2"
            shift 2
            ;;
        --results-dir)
            if [[ $# -lt 2 ]]; then
                echo "[error] --results-dir requires a path" >&2
                exit 1
            fi
            USER_RESULTS_DIR="$2"
            shift 2
            ;;
        --results-dir=*)
            USER_RESULTS_DIR="${1#--results-dir=}"
            shift
            ;;
        --dir-name)
            if [[ $# -lt 2 ]]; then
                echo "[error] --dir-name requires a value" >&2
                exit 1
            fi
            USER_DIR_NAME="$2"
            shift 2
            ;;
        --dir-name=*)
            USER_DIR_NAME="${1#--dir-name=}"
            shift
            ;;
        *)
            echo "[error] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done
if [[ -n "${TYPE}" ]]; then
    TYPE_ARGS=(--type "${TYPE}")
fi

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
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

# 结果基目录：默认 ./results/_single_module，可用环境变量 RESULTS_DIR 或命令行 --results-dir 指定
RESULTS_BASE="${RESULTS_DIR:-./results/_single_module}"
if [[ -n "${USER_RESULTS_DIR}" ]]; then
    RESULTS_BASE="${USER_RESULTS_DIR}"
fi
run_name="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${USER_DIR_NAME}" ]]; then
    RUN_DIR="${RESULTS_BASE}/${USER_DIR_NAME}"
else
    RUN_DIR="${RESULTS_BASE}/mutate_single_${run_name}"
fi
mkdir -p "${RUN_DIR}"

# 自定义目录名时路径已是最终 run 目录，single_module_mutate.py 不应再套一层 mutate_*
NO_SUBDIR_ARG=()
if [[ -n "${USER_DIR_NAME}" ]]; then
    NO_SUBDIR_ARG=(--no-create-results-subdir)
fi

# 清理可能遗留的 torch_extensions 锁文件，避免因上次崩溃导致本次一直卡在 "Using ... as PyTorch extensions root"
TORCH_EXT_CACHE="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}"
for lock in "${TORCH_EXT_CACHE}"/py310_cpu/*/lock; do
    [ -f "${lock}" ] && rm -f "${lock}" && echo "[env] removed stale lock: ${lock}"
done 2>/dev/null || true

export LMSV_MM_MSARUN=0
export LMSV_MM_PTARUN=1

torchrun ${DISTRIBUTED_ARGS} \
    single_module_mutate.py \
    ${GPT_ARGS} \
    --all-modules "${ALL_MODULES}" \
    --rounds "${ROUNDS}" \
    --iterations "${ITERATIONS}" \
    --results-dir "${RUN_DIR}" \
    "${NO_SUBDIR_ARG[@]}" \
    ${TYPE_ARGS[@]} \
    --distributed-backend nccl \
    2>&1 | tee "${RUN_DIR}/train.log"

chmod 440 "${RUN_DIR}/train.log"
