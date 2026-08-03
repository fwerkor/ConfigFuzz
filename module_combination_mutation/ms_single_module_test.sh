#!/bin/bash
# 基于单模块变异生成的配置文件目录或单个配置跑测：读取 configs 下或 --config 指定的 .json，
# 逐个用 SingleTextDecoderTemplate / SingleImageEncoderTemplate 实例化、前向 / Loss / 反向。
# 支持多卡、权重保存、基于配置/权重目录或单一配置/权重启动。
# 用法: ./single_module_test.sh [<配置文件目录>] [single_module_test.py args...]
# 例:   ./single_module_test.sh ./results/_single_module/mutate_single_20250319_120000/configs
# 例:   ./single_module_test.sh ./results/_single_module/mutate_single_xxx/configs --no-single-module-test-optimizer
# 例:   ./single_module_test.sh ./results/_single_module/mutate_single_xxx/configs --save-ckpt
# 例:   ./single_module_test.sh --config ./results/_single_module/mutate_single_xxx/configs/1-1-text_decoder.json
# 例:   ./single_module_test.sh --config .../configs/1-1-text_decoder.json --ckpt .../ckpts/test_single_xxx_001_1-1-text_decoder.pt
# msrun 日志目录：默认 ./msrun_log，可用 --msrun-log-dir <path> 或环境变量 MSRUN_LOG_DIR 指定。

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

if [ -z "${HCCL_IF_BASE_PORT}" ]; then
  export HCCL_IF_BASE_PORT=61000
fi
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

NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT=27111
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

# 兼容两种调用方式：
# 1) ./ms_single_module_test.sh <config_dir> [extra args...]
# 2) ./ms_single_module_test.sh --config <config.json> [extra args...]
#
# 额外支持脚本级参数：
#   --master-addr <addr> / --master-addr=<addr>
#   --node-rank <rank>   / --node-rank=<rank>
#   --nnodes <num>       / --nnodes=<num>
CONFIG_DIR=""
EXTRA_ARGS=()
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
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
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

MSRUN_LOG_DIR="${MSRUN_LOG_DIR:-msrun_log}"
declare -a _filtered_extra=()
_i=0
while (( _i < ${#EXTRA_ARGS[@]} )); do
  _a="${EXTRA_ARGS[$_i]}"
  if [[ "$_a" == "--msrun-log-dir" ]]; then
    _ni=$((_i + 1))
    if (( _ni < ${#EXTRA_ARGS[@]} )); then
      MSRUN_LOG_DIR="${EXTRA_ARGS[$_ni]}"
      _i=$((_i + 2))
      continue
    else
      echo "[arg] --msrun-log-dir 需要路径参数"
      exit 1
    fi
  elif [[ "$_a" == --msrun-log-dir=* ]]; then
    MSRUN_LOG_DIR="${_a#--msrun-log-dir=}"
    _i=$((_i + 1))
    continue
  fi
  _filtered_extra+=("$_a")
  _i=$((_i + 1))
done
EXTRA_ARGS=("${_filtered_extra[@]}")

TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
MBS="${MBS:-1}"
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
GBS=$(($WORLD_SIZE * $MBS / $CP))

DISTRIBUTED_ARGS="
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --join=True \
    --cluster_time_out=300 \
    --bind_core=True \
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
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --deterministic-mode \
    --npu-deterministic \
"
export NCCL_ALGO=Ring

USER_RESULTS_DIR=""
USER_CKPT_DIR=""
USER_CONFIG_DIR=""
USER_CONFIG_FILE=""
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  if [[ "$arg" == "--results-dir" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_ARGS[@]} )); then
      USER_RESULTS_DIR="${EXTRA_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --results-dir=* ]]; then
    USER_RESULTS_DIR="${arg#--results-dir=}"
  fi
  if [[ "$arg" == "--ckpt-dir" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_ARGS[@]} )); then
      USER_CKPT_DIR="${EXTRA_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --ckpt-dir=* ]]; then
    USER_CKPT_DIR="${arg#--ckpt-dir=}"
  fi
  if [[ "$arg" == "--config-dir" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_ARGS[@]} )); then
      USER_CONFIG_DIR="${EXTRA_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --config-dir=* ]]; then
    USER_CONFIG_DIR="${arg#--config-dir=}"
  fi
  if [[ "$arg" == "--config" ]]; then
    next_idx=$((i+1))
    if (( next_idx < ${#EXTRA_ARGS[@]} )); then
      USER_CONFIG_FILE="${EXTRA_ARGS[$next_idx]}"
    fi
  elif [[ "$arg" == --config=* ]]; then
    USER_CONFIG_FILE="${arg#--config=}"
  fi
done

INPUT_CONFIG_DIR="$USER_CONFIG_DIR"
if [[ -n "$USER_CONFIG_DIR" ]]; then
  INPUT_CONFIG_DIR="$USER_CONFIG_DIR"
elif [[ -n "$CONFIG_DIR" ]]; then
  INPUT_CONFIG_DIR="$CONFIG_DIR"
elif [[ -n "$USER_CONFIG_FILE" ]]; then
  if [[ "$USER_CONFIG_FILE" != */* ]]; then
    echo "[arg] --config 必须传入文件路径（不能只给文件名）：--config ${USER_CONFIG_FILE}"
    exit 1
  fi
  INPUT_CONFIG_DIR="$(dirname "$USER_CONFIG_FILE")"
fi

if [[ -z "$INPUT_CONFIG_DIR" ]]; then
  echo "Usage: $0 [config_dir] [single_module_test.py args...]"
  echo "  config_dir: directory containing single-module .json (e.g. ./results/_single_module/mutate_single_xxx/configs)"
  echo "  or:        $0 --config <config.json> [args...]"
  echo "  msrun 日志: --msrun-log-dir <dir> 或环境变量 MSRUN_LOG_DIR（默认: msrun_log，相对当前脚本目录）"
  echo "  extra args: --save-ckpt, --load-ckpt --load-ckpt-dir <dir>, --ckpt <file.pt>, --no-single-module-test-optimizer, etc."
  exit 1
fi

INPUT_CONFIG_DIR="$(cd -- "$INPUT_CONFIG_DIR" && pwd)"

if [[ -n "$USER_CONFIG_DIR" && ! -d "$INPUT_CONFIG_DIR" ]]; then
  echo "[arg] --config-dir 必须是存在的目录：--config-dir ${USER_CONFIG_DIR}"
  exit 1
fi

if [[ -n "$USER_CONFIG_FILE" ]]; then
  CFG_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$USER_CONFIG_FILE")"
  if [[ ! -f "$CFG_ABS" ]]; then
    echo "[arg] --config 必须是存在的 .json 文件：$CFG_ABS"
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
  MUTATE_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$USER_RESULTS_DIR")"
  PASS_RESULTS_DIR=0
else
  RESULTS_BASE="${RESULTS_DIR:-./results/_single_module}"
  run_name="test_single_$(date +%Y%m%d_%H%M%S)"
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
  CKPT_DIR="${RUN_DIR}/ckpts"
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
done 2>/dev/null || true

export LMSV_MM_MSARUN=1
export LMSV_MM_PTARUN=0

msrun $DISTRIBUTED_ARGS \
    --log_dir="$MSRUN_LOG_DIR" \
    single_module_test.py \
    $GPT_ARGS \
    $( (( PASS_CONFIG_DIR )) && printf '%s ' --config-dir "${MUTATE_DIR}/configs" ) \
    $( (( PASS_RESULTS_DIR )) && printf '%s ' --results-dir "${MUTATE_DIR}" ) \
    $( (( PASS_CKPT_DIR )) && printf '%s ' --ckpt-dir "${CKPT_DIR}" ) \
    --distributed-backend nccl \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${MUTATE_DIR}/single_module_test.log"
chmod 440 "${MUTATE_DIR}/single_module_test.log" 2>/dev/null || true
