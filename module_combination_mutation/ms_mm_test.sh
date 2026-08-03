#!/bin/bash
# 基于配置文件目录跑测：读取目录下所有 .json 配置，逐个初始化 template、前向 / Loss / 反向
# 用法: ./ms_mm_test.sh [<配置文件目录>] [mm_test.py args...]
# msrun 日志目录：环境变量 MSRUN_LOG_DIR，或参数 --msrun-log-dir <path>（默认 ../lmsv_rec/msrun_log，相对脚本目录）
# 例:   ./ms_mm_test.sh ./results/run_20260307_230243/configs
# 例:   ./ms_mm_test.sh ./results/run_20260307_230243/configs --no-mm-test-optimizer
# 例:   ./ms_mm_test.sh ./results/run_20260307_230243/configs --save-ckpt   # 保存权重到 <test_run>/ckpts/
# 例:   ./ms_mm_test.sh --config ./results/run_20260307_230243/configs/round_0.json
# 例:   ./ms_mm_test.sh ./configs --msrun-log-dir ./my_msrun_log

set -e
set -o pipefail
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_FUSION_OP_ENABLE=0
export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-$SCRIPT_DIR/cache_ascend}"

# HCCL 通信需占用一段连续端口，未预留时易报 "port already bound" (error code 7)。
# 默认使用 60000-60015；若已占用可设置 HCCL_IF_BASE_PORT 为其它起始端口并预留该起 16 个端口。
if [ -z "${HCCL_IF_BASE_PORT}" ]; then
  export HCCL_IF_BASE_PORT=60000
fi
# 预留 HCCL 端口（修改 sysctl 必须 root）。先直接试，再尝试 sudo，否则需用户手动执行一次。
RESERVED_PORTS="${HCCL_IF_BASE_PORT}-$((${HCCL_IF_BASE_PORT} + 15))"
if sysctl -w net.ipv4.ip_local_reserved_ports="${RESERVED_PORTS}" 2>/dev/null; then
  echo "[HCCL] 已预留端口 ${RESERVED_PORTS}"
elif sudo -n sysctl -w net.ipv4.ip_local_reserved_ports="${RESERVED_PORTS}" 2>/dev/null; then
  echo "[HCCL] 已预留端口 ${RESERVED_PORTS} (sudo)"
else
  echo "[HCCL] 正在用 sudo 预留端口 ${RESERVED_PORTS}（可能需要输入密码）..."
  if sudo sysctl -w net.ipv4.ip_local_reserved_ports="${RESERVED_PORTS}"; then
    echo "[HCCL] 已预留端口 ${RESERVED_PORTS}"
  else
    echo "[HCCL] 预留失败。若多卡报错 port already bound，请手动执行："
    echo "  sudo sysctl -w net.ipv4.ip_local_reserved_ports=${RESERVED_PORTS}"
  fi
fi

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=28959
NNODES=1
NODE_RANK=0

TP=1
PP=1
CP=1
MBS=1

export NCCL_ALGO=Ring
# 兼容两种调用方式：
# 1) ./mm_test.sh <config_dir> [extra args...]
# 2) ./mm_test.sh --config <config.json> [extra args...]   （不提供位置参数）
#
# 额外支持脚本级参数：
#   --master-addr <addr> / --master-addr=<addr>
#   --node-rank <rank>   / --node-rank=<rank>
#   --nnodes <num>       / --nnodes=<num>
CONFIG_DIR=""
EXTRA_MM_TEST_ARGS=()
USER_SET_MASTER_ADDR=0
USER_SET_NODE_RANK=0
USER_SET_NNODES=0

RAW_ARGS=("$@")
if [[ ${#RAW_ARGS[@]} -gt 0 && "${RAW_ARGS[0]}" != -* ]]; then
  CONFIG_DIR="${RAW_ARGS[0]}"
  set -- "${RAW_ARGS[@]:1}"
else
  set -- "${RAW_ARGS[@]}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --master-addr)
      if [[ $# -lt 2 ]]; then
        echo "[arg] --master-addr 需要一个参数值"
        exit 1
      fi
      MASTER_ADDR="$2"
      USER_SET_MASTER_ADDR=1
      shift 2
      ;;
    --master-addr=*)
      MASTER_ADDR="${1#--master-addr=}"
      USER_SET_MASTER_ADDR=1
      shift
      ;;
    --node-rank)
      if [[ $# -lt 2 ]]; then
        echo "[arg] --node-rank 需要一个参数值"
        exit 1
      fi
      NODE_RANK="$2"
      USER_SET_NODE_RANK=1
      shift 2
      ;;
    --node-rank=*)
      NODE_RANK="${1#--node-rank=}"
      USER_SET_NODE_RANK=1
      shift
      ;;
    --nnodes)
      if [[ $# -lt 2 ]]; then
        echo "[arg] --nnodes 需要一个参数值"
        exit 1
      fi
      NNODES="$2"
      USER_SET_NNODES=1
      shift 2
      ;;
    --nnodes=*)
      NNODES="${1#--nnodes=}"
      USER_SET_NNODES=1
      shift
      ;;
    --)
      shift
      EXTRA_MM_TEST_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_MM_TEST_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || (( NNODES < 1 )); then
  echo "[arg] --nnodes 必须是 >= 1 的整数，当前值：$NNODES"
  exit 1
fi
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK < 0 )); then
  echo "[arg] --node-rank 必须是 >= 0 的整数，当前值：$NODE_RANK"
  exit 1
fi

if (( USER_SET_MASTER_ADDR && USER_SET_NODE_RANK && USER_SET_NNODES )); then
  export MA_NUM_HOSTS="$NNODES"
  export VC_TASK_INDEX="$NODE_RANK"
  export MASTER_ADDR="$MASTER_ADDR"
fi

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
GBS=$(($WORLD_SIZE*$MBS/$CP))

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
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --retro-encoder-attention-dropout 0.0 \
    --retro-encoder-hidden-dropout 0.0 \
    --deterministic-mode \
    --npu-deterministic \
"

# 若用户在 extra args 中显式指定了各种 dir，则使用用户提供的，否则使用脚本默认值。
USER_RESULTS_DIR=""
USER_CKPT_DIR=""
USER_CONFIG_DIR=""
USER_CONFIG_FILE=""
CLI_MSRUN_LOG_DIR=""
for ((i=0; i<${#EXTRA_MM_TEST_ARGS[@]}; i++)); do
  arg="${EXTRA_MM_TEST_ARGS[$i]}"

  if [[ "$arg" == "--msrun-log-dir" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_MM_TEST_ARGS[@]} )); then
      CLI_MSRUN_LOG_DIR="${EXTRA_MM_TEST_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --msrun-log-dir=* ]]; then
    CLI_MSRUN_LOG_DIR="${arg#--msrun-log-dir=}"
  fi

  if [[ "$arg" == "--results-dir" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_MM_TEST_ARGS[@]} )); then
      USER_RESULTS_DIR="${EXTRA_MM_TEST_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --results-dir=* ]]; then
    USER_RESULTS_DIR="${arg#--results-dir=}"
  fi

  if [[ "$arg" == "--ckpt-dir" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_MM_TEST_ARGS[@]} )); then
      USER_CKPT_DIR="${EXTRA_MM_TEST_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --ckpt-dir=* ]]; then
    USER_CKPT_DIR="${arg#--ckpt-dir=}"
  fi

  if [[ "$arg" == "--config-dir" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_MM_TEST_ARGS[@]} )); then
      USER_CONFIG_DIR="${EXTRA_MM_TEST_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --config-dir=* ]]; then
    USER_CONFIG_DIR="${arg#--config-dir=}"
  fi

  if [[ "$arg" == "--config" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_MM_TEST_ARGS[@]} )); then
      USER_CONFIG_FILE="${EXTRA_MM_TEST_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --config=* ]]; then
    USER_CONFIG_FILE="${arg#--config=}"
  fi
done

FILTERED_MM_TEST_ARGS=()
skip_next=0
for ((i=0; i<${#EXTRA_MM_TEST_ARGS[@]}; i++)); do
  if (( skip_next )); then
    skip_next=0
    continue
  fi
  arg="${EXTRA_MM_TEST_ARGS[$i]}"
  if [[ "$arg" == "--msrun-log-dir" ]]; then
    skip_next=1
    continue
  fi
  if [[ "$arg" == --msrun-log-dir=* ]]; then
    continue
  fi
  FILTERED_MM_TEST_ARGS+=("$arg")
done

MSRUN_LOG_DIR="${CLI_MSRUN_LOG_DIR:-$MSRUN_LOG_DIR}"
MSRUN_LOG_DIR="${MSRUN_LOG_DIR:-$SCRIPT_DIR/msrun_log}"
mkdir -p "$MSRUN_LOG_DIR"
MSRUN_LOG_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$MSRUN_LOG_DIR")"

DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --log_dir=msrun_log \
    --join=True \
    --cluster_time_out=300 \
    --bind_core=True
"

# 输入 config_dir：优先使用 --config-dir；其次位置参数；若都没有则尝试从 --config 推导
INPUT_CONFIG_DIR="$USER_CONFIG_DIR"
if [[ -n "$USER_CONFIG_DIR" ]]; then
  INPUT_CONFIG_DIR="$USER_CONFIG_DIR"
elif [[ -n "$CONFIG_DIR" ]]; then
  INPUT_CONFIG_DIR="$CONFIG_DIR"
elif [[ -n "$USER_CONFIG_FILE" ]]; then
  # 仅在 --config 给的是路径时可推导目录；若只是文件名则无法推导
  if [[ "$USER_CONFIG_FILE" != */* ]]; then
    echo "[arg] --config 必须传入文件路径（不能只给文件名）：--config ${USER_CONFIG_FILE}"
    exit 1
  fi
  INPUT_CONFIG_DIR="$(dirname "$USER_CONFIG_FILE")"
fi

if [[ -z "$INPUT_CONFIG_DIR" ]]; then
  echo "Usage: $0 [config_dir] [mm_test.py args...]"
  echo "  config_dir: directory containing round_*.json (e.g. ./results/run_20260307_230243/configs)"
  echo "  or:        $0 --config <config.json> [mm_test.py args...]"
  echo "  extra args : forwarded to mm_test.py (e.g. --no-mm-test-optimizer)"
  echo "  --msrun-log-dir <path> : msrun worker/scheduler logs (default: ../lmsv_rec/msrun_log; env: MSRUN_LOG_DIR)"
  exit 1
fi

INPUT_CONFIG_DIR="$(cd -- "$INPUT_CONFIG_DIR" && pwd)"

# 参数约束：--config-dir 只能是目录路径；--config 只能是文件路径
if [[ -n "$USER_CONFIG_DIR" && ! -d "$INPUT_CONFIG_DIR" ]]; then
  echo "[arg] --config-dir 必须是存在的目录路径：--config-dir ${USER_CONFIG_DIR}"
  exit 1
fi

if [[ -n "$USER_CONFIG_FILE" ]]; then
  CFG_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$USER_CONFIG_FILE")"
  if [[ ! -f "$CFG_ABS" ]]; then
    echo "[arg] --config 必须是存在的 .json 文件路径：$CFG_ABS"
    exit 1
  fi
  if [[ "$CFG_ABS" != *.json ]]; then
    echo "[arg] --config 必须是 .json 文件：$CFG_ABS"
    exit 1
  fi
fi

PASS_RESULTS_DIR=1
PASS_CKPT_DIR=1
if [[ -n "$USER_RESULTS_DIR" ]]; then
  # 用户指定了 results-dir：不再强行写到 test_xxx/<mutate_name>/ 下
  MUTATE_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$USER_RESULTS_DIR")"
  PASS_RESULTS_DIR=0
else
  # 将本次跑测的输出统一放在：<RUN_DIR>/<mutate_name>/
  # 约定：如果传入的是 .../mutate_xxx/configs，则 mutate_name=mutate_xxx（而不是 configs）
  # 并将输入 configs 复制到 <mutate_name>/configs，便于完整复现。
  RESULTS_BASE="${RESULTS_DIR:-./results/_module_combine}"
  run_name="test_$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="${RESULTS_BASE}/${run_name}"
  mkdir -p "${RUN_DIR}"

  CONFIG_BASENAME="$(basename "$INPUT_CONFIG_DIR")"
  if [ "$CONFIG_BASENAME" = "configs" ]; then
    MUTATE_NAME="$(basename "$(dirname "$INPUT_CONFIG_DIR")")"
  else
    MUTATE_NAME="$CONFIG_BASENAME"
  fi
  MUTATE_DIR="${RUN_DIR}/${MUTATE_NAME}"
fi

if [[ -n "$USER_CKPT_DIR" ]]; then
  CKPT_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$USER_CKPT_DIR")"
  PASS_CKPT_DIR=0
else
  # 默认 ckpt 放在 run 级别（与 mm_test.py 的说明一致，避免强制嵌套到 results-dir 下）
  CKPT_DIR="${RUN_DIR}/ckpts"
  # 若用户指定了 results-dir，但未指定 ckpt-dir，则把 ckpt 放到 results-dir/ckpts 更直观
  if (( ! PASS_RESULTS_DIR )); then
    CKPT_DIR="${MUTATE_DIR}/ckpts"
  fi
fi

mkdir -p "${MUTATE_DIR}"
mkdir -p "${CKPT_DIR}"

PASS_CONFIG_DIR=1
if [[ -n "$USER_CONFIG_DIR" ]]; then
  PASS_CONFIG_DIR=0
else
  rm -rf "${MUTATE_DIR}/configs" 2>/dev/null || true
  cp -r "${INPUT_CONFIG_DIR}" "${MUTATE_DIR}/configs"
fi

TORCH_EXT_CACHE="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}"
for lock in "${TORCH_EXT_CACHE}"/py310_cpu/*/lock; do
  [ -f "$lock" ] && rm -f "$lock" && echo "[env] removed stale lock: $lock"
done 2>/dev/null

export LMSV_MM_MSARUN=1
export LMSV_MM_PTARUN=0

msrun $DISTRIBUTED_ARGS \
    mm_test.py \
    $GPT_ARGS \
    $( (( PASS_CONFIG_DIR )) && printf '%s ' --config-dir "${MUTATE_DIR}/configs" ) \
    $( (( PASS_RESULTS_DIR )) && printf '%s ' --results-dir "${MUTATE_DIR}" ) \
    $( (( PASS_CKPT_DIR )) && printf '%s ' --ckpt-dir "${CKPT_DIR}" ) \
    --distributed-backend nccl \
    "${FILTERED_MM_TEST_ARGS[@]}" \
    2>&1 | tee "${MUTATE_DIR}/mm_test.log"
chmod 440 "${MUTATE_DIR}/mm_test.log" 2>/dev/null || true
