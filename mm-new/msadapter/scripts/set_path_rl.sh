#!/bin/bash
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH

script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
msadapter_dir=$(dirname $script_dir)
MindSpeed_Core_MS_PATH=${msadapter_dir}/MindSpeed-Core-MS
rm -rf ${MindSpeed_Core_MS_PATH}/RL/msadapter
export PYTHONPATH=${msadapter_dir}/msadapter/:${msadapter_dir}/msadapter/msa_thirdparty/:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/RL/Megatron-LM/:${MindSpeed_Core_MS_PATH}/RL/MindSpeed/:${MindSpeed_Core_MS_PATH}/RL/MindSpeed-LLM/:${MindSpeed_Core_MS_PATH}/RL/MindSpeed-RL/:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/RL/transformers/src:${MindSpeed_Core_MS_PATH}/RL/vllm/:${MindSpeed_Core_MS_PATH}/RL/vllm-ascend/:$PYTHONPATH
echo "..............................................done set RL_PYTHONPATH"
echo $PYTHONPATH
