#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
msadapter_dir=$(dirname $script_dir)
MindSpeed_Core_MS_PATH=${msadapter_dir}/MindSpeed-Core-MS
rm -rf ${msadapter_dir}/MindSpeed-Core-MS/msadapter
export PYTHONPATH=${msadapter_dir}/msadapter/:${msadapter_dir}/msadapter/msa_thirdparty/:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/:${MindSpeed_Core_MS_PATH}/Megatron-LM/:${MindSpeed_Core_MS_PATH}/MindSpeed/:$PYTHONPATH
echo "..............................................done set PYTHONPATH"
echo $PYTHONPATH