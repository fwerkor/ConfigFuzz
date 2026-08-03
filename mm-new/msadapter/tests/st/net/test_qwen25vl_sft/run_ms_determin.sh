#!/bin/bash

#安装依赖环境
# pip install -r requirements.txt

pip install torchvision==0.16.0

MindSpeed_Core_MS_PATH=../../../../MindSpeed-Core-MS/MM
MindSpeed_MM_PATH=../../../../MindSpeed-Core-MS/MM/MindSpeed-MM
dataset_path="/home/workspace/mindspore_dataset/msadapter/test_input/net/test_qwen25vl_sft_lora"
data_src="${MindSpeed_MM_PATH}/examples/mindspore/qwen2.5vl/data_72b.json"
model_src="${MindSpeed_MM_PATH}/examples/mindspore/qwen2.5vl/model_72b.json"

backup() {
    fname=$1
    cp $fname $fname'_back'
    echo '======'$fname 'backuped!'
}

recover() {
    fname=$1
    cp $fname'_back' $fname
    echo '======'$fname 'recovered!!!!'
}

memRecord() {
    recordFile=$1
    bash mem.sh $recordFile > mem.txt 2>&1&
}

addSeedAll() {
    fname=$1
    lineNumMain=$(grep -n '__main__' ${fname} | cut -d: -f1)
    echo deterministic
    sed -i $((lineNumMain + 1))'i\ \ \ \ seed_all()' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch_npu.npu.manual_seed(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch_npu.npu.manual_seed_all(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch.use_deterministic_algorithms(True)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch.manual_seed(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ np.random.seed(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ os.environ["PYTHONHASHSEED"] = str(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ random.seed(seed)' $fname
    sed -i $((lineNumMain - 1))'idef seed_all(seed=42):' $fname
    sed -i $((lineNumMain - 1))'iimport torch_npu' $fname
    sed -i $((lineNumMain - 1))'iimport torch' $fname
    sed -i $((lineNumMain - 1))'iimport numpy as np' $fname
    sed -i $((lineNumMain - 1))'iimport random' $fname
    sed -i $((lineNumMain - 1))'iimport os' $fname
}

# 打印16位日志
modifyTrainingLogs() {
    fname=$1
    echo "Modifying training log precision..."
    # 替换 log_string += ' {}: {:.6E} |'.format(key, avg)
    sed -i 's/log_string += '\'' {}: {:.6E} |'\''.format(key, avg)/log_string += '\'' {}: {:.16f} |'\''.format(key, avg)/g' "$fname"
    # 替换 log_string += ' grad norm: {:.3f} |'.format(grad_norm)
    sed -i 's/log_string += '\'' grad norm: {:.3f} |'\''.format(grad_norm)/log_string += '\'' grad norm: {:.16f} |'\''.format(grad_norm)/g' "$fname"
    # 替换 log_string += ' params norm: {:.3f} |'.format(params_norm)
    sed -i 's/log_string += '\'' params norm: {:.3f} |'\''.format(params_norm)/log_string += '\'' params norm: {:.16f} |'\''.format(params_norm)/g' "$fname"
    echo "Log precision has been updated to 16 decimal places in $fname"
}


# 开确定性计算跑一遍
backup ${MindSpeed_MM_PATH}/pretrain_vlm.py
backup ${MindSpeed_MM_PATH}/mindspeed_mm/training.py
backup ${data_src}
backup ${model_src}
backup ${MindSpeed_MM_PATH}/mindspeed_mm/data/data_utils/video_processor.py
addSeedAll ${MindSpeed_MM_PATH}/pretrain_vlm.py
modifyTrainingLogs ${MindSpeed_MM_PATH}/mindspeed_mm/training.py
export HCCL_DETERMINISTIC=true  # HCCL确定性
export ASCEND_LAUNCH_BLOCKING=1  # 硬件确定性
export NCCL_DETERMINISTIC=1
export CLOSE_MATMUL_K_SHIFT=1 # 设置matmul行为

#修改data_72b.json
sed -i 's|"./ckpt/hf_path/Qwen2.5-VL-72B-Instruct"|"'$dataset_path'/Qwen2.5-VL-72B-Instruct"|g' $data_src
sed -i 's|"./data"|"'$dataset_path'/datasets"|g' $data_src
sed -i 's|"./data/mllm_format_llava_instruct_data.json"|"'$dataset_path'/datasets/mllm_format_llava_instruct_data.json"|g' $data_src
sed -i 's|"./data/cache_dir"|"'$MindSpeed_MM_PATH'/cache_dir_ms_lora_72b_100"|g' $data_src
sed -i 's|"preprocessing_batch_size": 1000|"preprocessing_batch_size": 100|g' $data_src
sed -i 's|"preprocessing_num_workers": 16|"preprocessing_num_workers": 1|g' $data_src
sed -i 's|"max_samples": null|"max_samples": 100|g' $data_src
sed -i 's|"shuffle": true|"shuffle": false|g' $data_src

#修改model_72b.json
sed -i 's|"num_layers": 32|"num_layers": 2|g' $model_src
sed -i 's|"pipeline_num_layers": \[32, 0, 0, 0, 0, 0, 0, 0\],|"pipeline_num_layers": [2, 0],|g' $model_src
sed -i 's|"num_layers": 80|"num_layers": 2|g' $model_src
sed -i 's|"pipeline_num_layers": \[6, 11, 11, 11, 11, 11, 11, 8\]|"pipeline_num_layers": [1, 1]|g' $model_src

#video_processor.py
sed -i 's@video_fps: int | float,@video_fps,@g' ${MindSpeed_MM_PATH}/mindspeed_mm/data/data_utils/video_processor.py


#heterogeneous_config.py
sed -i 's@num_query_groups: int | None = None@num_query_groups = None@g' ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/core/transformer/heterogeneous/heterogeneous_config.py
sed -i 's@ffn_hidden_size: float | None = None@ffn_hidden_size = None@g' ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/core/transformer/heterogeneous/heterogeneous_config.py

bash test_qwen25vl_sft_ms.sh > ms_det.txt

cat ms_det.txt
recover ${MindSpeed_MM_PATH}/pretrain_vlm.py
recover ${MindSpeed_MM_PATH}/mindspeed_mm/training.py
recover ${data_src}
recover ${model_src}
recover ${MindSpeed_MM_PATH}/mindspeed_mm/data/data_utils/video_processor.py
pip uninstall -y torchvision
pip uninstall -y torch