#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

source ../../../../scripts/set_path_mm.sh
MindSpeed_MM_PATH=../../../../MindSpeed-Core-MS/MM/MindSpeed-MM

NPUS_PER_NODE=8
export MASTER_ADDR=localhost
MASTER_PORT=6981
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
export LOCAL_WORLD_SIZE=$NPUS_PER_NODE


MM_DATA="${MindSpeed_MM_PATH}/examples/mindspore/qwen2.5vl/data_72b.json"
MM_MODEL="${MindSpeed_MM_PATH}/examples/mindspore/qwen2.5vl/model_72b.json"
MM_TOOL="${MindSpeed_MM_PATH}/mindspeed_mm/tools/tools.json"
LOAD_PATH="/home/workspace/mindspore_dataset/msadapter/test_input/net/test_qwen25vl_sft/ckpt_model"
SAVE_PATH="/home/workspace/mindspore_dataset/msadapter/test_input/net/test_qwen25vl_sft/ckpt_model"

TP=2
PP=2
CP=1
MBS=1
GRAD_ACC_STEP=4
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

DISTRIBUTED_ARGS="
    --local_worker_num $NPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --log_dir msrun_log \
    --bind_core=True \
    --join=True 
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --tokenizer-type NullTokenizer \
    --vocab-size 152064 \
    --seq-length 1024 \
    --make-vocab-size-divisible-by 1 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 0 \
    --train-iters 10 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --seed 42 \
    --bf16 \
    --variable-seq-lengths \
    --use-distributed-optimizer \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 8 \
    --use-flash-attn \
"

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 5000 \
    --ckpt-format torch \
"
    # --load $LOAD_PATH \
    # --save $SAVE_PATH \

logfile=train_$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
msrun $DISTRIBUTED_ARGS ${MindSpeed_MM_PATH}/pretrain_vlm.py \
    $GPT_ARGS \
    $LORA_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --ai-framework mindspore \
    | tee qwen25vl_sft.log