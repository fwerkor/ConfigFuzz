#!/bin/bash

rm -rf MindSpeed-Core-MS/
git clone -b feature-0.2 https://gitee.com/ascend/MindSpeed-Core-MS.git
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-Core-MS"
    exit 1
fi

cd MindSpeed-Core-MS/
rm -rf tests/
rm -rf MindSpeed-LLM/
git clone https://gitee.com/ascend/MindSpeed-LLM.git
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-LLM"
    exit 1
fi

cd MindSpeed-LLM
git checkout 84a34ee5e10da1ef5beff0787e94605aa961d3ad
cd ..
echo "------------------------------------done MindSpeed-LLM"


rm -rf MindSpeed/
git clone https://gitee.com/ascend/MindSpeed.git
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed"
    exit 1
fi
cd MindSpeed
git checkout 0dfa0035ec54d9a74b2f6ee2867367df897299df
cd ..
echo "...............................................done MindSpeed"

#Megatron-LM
rm -rf Megatron-LM/
git clone https://gitee.com/mirrors/Megatron-LM.git
if [ $? -ne 0 ]; then
    echo "Error: git clone Megatron-LM"
    exit 1
fi
cd Megatron-LM
git checkout core_r0.8.0
cd ..
echo "..............................................done Megatron-LM"


#transformers
rm -rf transformers/
git clone https://gitee.com/mirrors/huggingface_transformers.git -b v4.47.0
if [ $? -ne 0 ]; then
    echo "Error: git clone huggingface_transformers"
    exit 1
fi
mv huggingface_transformers transformers
cd transformers
git apply ../tools/rules/transformers.diff
cd ..
echo "..............................................done apply transformers"

echo "..............................................start code_convert"
MindSpeed_Core_MS_PATH=$PWD
echo ${MindSpeed_Core_MS_PATH}

python3 tools/transfer.py \
--megatron_path ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/ \
--mindspeed_path ${MindSpeed_Core_MS_PATH}/MindSpeed/mindspeed/ \
--mindspeed_llm_path ${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/


echo "..............................................done code_convert"

#install pkg
cd ..
pip install -r ./.jenkins/test/config/requirements.txt

pip uninstall --yes torch torch-npu torchvision accelerate

export PYTHONPATH=${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
echo "Please verify your PYTHONPATH: "$PYTHONPATH