#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
msadapter_dir=$(dirname $script_dir)
MindSpeed_Core_MS_PATH=${msadapter_dir}/MindSpeed-Core-MS/MM

cd ${MindSpeed_Core_MS_PATH}
echo "-------------------------set mm commit_id-------------------------"
cd MindSpeed
git checkout 2846b4cc45f5750a4e94c847254212ee790235cf #1030
cd ..
rm -rf MindSpeed/tests_extend
echo "------------------------------------done set MindSpeed commit_id"

cd MindSpeed-MM
git checkout b2f5d380d4827af170ad23aa79cef71cd62f67c8 #1101
cd ..
rm -rf MindSpeed-MM/tests
echo "------------------------------------done set MindSpeed-MM commit_id"
echo "-------------------------done set mm commit_id-------------------------"