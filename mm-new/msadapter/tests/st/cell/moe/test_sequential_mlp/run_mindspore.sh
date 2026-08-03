#!/bin/bash

set -e
LAYOUT=${1:-"SBH"}
export ASCEND_RT_VISIBLE_DEVICES=${2:-"0"}
tensor_parallel=${3:-"1"}
data_dir=${4:-"data/random_data/"}
ckpt_dir=${5:-"data/random_ckpt/"}
output_dir=${6:-"data/output/"}
log_path=${7:-"msrun_log"}

project_dir=$(dirname "$0")
echo "in project_dir: $project_dir"

RUN_TYPE="
    --run_mode test_mindspore \
    --layout $LAYOUT \
    --ckpt_dir "$ckpt_dir" \
    --data_dir "$data_dir" \
    --output_dir "$output_dir" \
"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0
export HCCL_BUFFSIZE=200
export HCCL_EXEC_TIMEOUT=600
export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1

export NCCL_DETERMINISTIC=1


MASTER_PORT=7415

PIDS=$(lsof -i :$MASTER_PORT | awk 'NR>1 {print $2}')
if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        kill -9 $pid
        echo "Killed process $pid"
    done
else
    echo "No processes found listening on port $MASTER_PORT."
fi

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


DISTRIBUTED_ARGS="
    --worker_num "$work_num" \
    --local_worker_num="$work_num" \
    --cluster_time_out=300 \
    --log_dir="$log_path" \
    --master_port=$MASTER_PORT
    --join=True \
"

MODEL_ARGS="
    --num-layers 4 \
    --seq-length 4096 \
    --vocab-size 256 \
    --hidden-size 128 \
    --ffn-hidden-size 2 \
    --num-attention-heads 16 \
    --max-position-embeddings 4096 \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --swiglu \
    --no-gradient-accumulation-fusion \
    --tokenizer-type NullTokenizer \
    --use-mcore-models \
    --disable-bias-linear \
    --bf16 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
"

MOE_ARGS="
    --moe-tp-extend-ep \
    --moe-token-dispatcher-type alltoall \
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 64 \
    --moe-router-topk 6 \
    --moe-intermediate-size 2048 \
    --moe-router-load-balancing-type noaux_tc \
    --topk-group 2 \
    --routed-scaling-factor 1.0 \
    --seq-aux \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
"

DATA_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --mock-data \
"

# if use sp, add '--sequence-parallel' args
PARALLEL_ARGS="
    --tensor-model-parallel-size $tensor_parallel \
    --pipeline-model-parallel-size 2 \
    --expert-model-parallel-size 4 \
    --sequence-parallel \
"

msrun $DISTRIBUTED_ARGS $project_dir/run_sequential_mlp.py \
      $RUN_TYPE \
      $MODEL_ARGS \
      $DATA_ARGS \
      $PARALLEL_ARGS \
      $MOE_ARGS
