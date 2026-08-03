#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
msadapter_dir=$(dirname $script_dir)
MindSpeed_Core_MS_PATH=${msadapter_dir}/MindSpeed-Core-MS

cd ${MindSpeed_Core_MS_PATH}
echo "-------------------------set llm commit_id-------------------------"
cd MindSpeed
git checkout 46f502ad90a942361d0f7b9fc94bf3fd19e2ac80 #0929
cd ..
rm -rf MindSpeed/tests_extend
echo "------------------------------------done set MindSpeed commit_id"

cd MindSpeed-LLM
git checkout a23caf6ed7fa065c67e602afae6763c65be47f91 #0929
cd ..
rm -rf MindSpeed-LLM/tests
echo "------------------------------------done set MindSpeed-LLM commit_id"
echo "-------------------------done set llm commit_id--------------- ----------"