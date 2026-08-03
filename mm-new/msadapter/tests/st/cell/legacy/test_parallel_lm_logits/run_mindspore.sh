#!/bin/bash

set -e

LAYOUT=${1:-"SBH"}
export ASCEND_RT_VISIBLE_DEVICES=${2:-"0"}
tensor_parallel=${3:-"1"}
data_dir=${4:-"data/alone/random_data/"}
ckpt_dir=${5:-"data/alone/random_ckpt/"}
output_dir=${6:-"data/alone/output/"}
log_path=${7:-"msrun_log"}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0
export HCCL_BUFFSIZE=200
export HCCL_EXEC_TIMEOUT=600
export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1
export USE_MS_BACKEND=1

MASTER_PORT=8418
# kill process
PIDS=$(lsof -i :$MASTER_PORT | awk 'NR>1 {print $2}')
if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        kill -9 $pid
        echo "Killed process $pid"
    done
else
    echo "No processes found listening on MASTER_PORT $MASTER_PORT."
fi

project_dir=$(dirname "$0")
echo "in project_dir: $project_dir"

dirs=(${data_dir}
      ${ckpt_dir}
      ${output_dir}/mindspore_forward
      ${output_dir}/mindspore_backward)
for dir in ${dirs[@]}; do
    mkdir -p $dir
done

rm -rf "${log_path}"
mkdir "${log_path}"
echo "train start, log path: ${log_path}"

# 计算设备数量
IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
work_num=${#devices[@]}
export WORLD_SIZE=$work_num

MODEL_ARGS="
    --num-layers 8 \
    --seq-length 4096 \
    --vocab-size 256 \
    --hidden-size 6144 \
    --ffn-hidden-size 1 \
    --num-attention-heads 32 \
    --max-position-embeddings 4096 \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --no-gradient-accumulation-fusion \
    --tokenizer-type NullTokenizer \
    --bf16\
"

DATA_ARGS="
    --micro-batch-size 1 \
    --mock-data \
"

# if use sp, add '--sequence-parallel' args
PARALLEL_ARGS="
    --tensor-model-parallel-size $tensor_parallel \
    --pipeline-model-parallel-size 1 \
"

msrun --worker_num "$work_num" \
      --local_worker_num="$work_num" \
      --master_port=$MASTER_PORT --log_dir="$log_path" \
      --join=True \
      --cluster_time_out=300 \
      $project_dir/run_parallel_lm_logits.py \
      --run_mode test_mindspore \
      --layout $LAYOUT \
      --ckpt_dir "$ckpt_dir" \
      --data_dir "$data_dir" \
      --output_dir "$output_dir" \
      $MODEL_ARGS \
      $DATA_ARGS \
      $PARALLEL_ARGS
