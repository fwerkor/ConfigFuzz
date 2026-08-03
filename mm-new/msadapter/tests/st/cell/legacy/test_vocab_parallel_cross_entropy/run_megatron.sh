#!/bin/bash

# set load path
LAYOUT=${1:-"SBH"}
export ASCEND_RT_VISIBLE_DEVICES=${2:-"0"}
tensor_parallel=${3:-"1"}
data_dir=${4:-"data/alone/random_data/"}
ckpt_dir=${5:-"data/alone/random_ckpt/"}
output_dir=${6:-"data/alone/output/"}
LOG_FILE=${7:-"megatron_alone.log"}

project_dir=$(dirname "$0")

RUN_TYPE="
    --run_mode test_megatron \
    --layout $LAYOUT \
    --ckpt_dir "$ckpt_dir" \
    --data_dir "$data_dir" \
    --output_dir "$output_dir" \
"

# visible ascend id of device
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0
export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1

# calculate num of device
IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
NPUS_PER_NODE=${#devices[@]}
MASTER_ADDR=localhost
MASTER_PORT=8417
NNODES=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

PIDS=$(lsof -i :$MASTER_PORT | awk 'NR>1 {print $2}')
if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        kill -9 $pid
        echo "Killed process $pid"
    done
else
    echo "No processes found listening on port $port."
fi

dirs=(${data_dir}
      ${ckpt_dir}
      ${output_dir}/megatron_forward
      ${output_dir}/megatron_backward)
for dir in ${dirs[@]}; do
    mkdir -p $dir
done


DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL_ARGS="
    --num-layers 8 \
    --seq-length 64 \
    --vocab-size 512 \
    --hidden-size 128 \
    --ffn-hidden-size 512 \
    --num-attention-heads 32 \
    --max-position-embeddings 4096 \
    --make-vocab-size-divisible-by 128 \
    --untie-embeddings-and-output-weights \
    --normalization RMSNorm \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --no-load-optim \
    --no-load-rng \
    --no-gradient-accumulation-fusion \
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

torchrun $DISTRIBUTED_ARGS $project_dir/run_vocab_parallel_cross_entropy.py \
    $RUN_TYPE \
    $MODEL_ARGS \
    $DATA_ARGS \
    $PARALLEL_ARGS \
    --distributed-backend nccl >& $LOG_FILE

