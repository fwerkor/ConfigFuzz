# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test ParallelColumnLinear"""
import os
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
from utils_st import compare_all_data, init_env


class TestParallelColumnLinear:

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.level0
    @pytest.mark.run(order=2)
    def test_mindspore_parallel_column_linear(self):
        """
        Feature: test mindspore parallel column linear.
        Description: run mindspore parallel column linear to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_mindspore.sh"
        layout = "SBH"
        devices = "0,1,2,3,4,5,6,7"
        tensor_parallel = 4
        log_dir = "msrun_parallel_log"

        init_env(mode='mindspore')
        print("os.environ['PYTHONPATH']:", os.environ['PYTHONPATH'])
        print("sys.path:", sys.path)
        print("os.environ:", os.environ)
        dir_path = os.path.split(os.path.realpath(__file__))[0]
        tests_path = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))
        test_input = "/home/workspace/mindspore_dataset/msadapter/test_input"
        test_path = os.path.join(test_input, os.path.relpath(dir_path, tests_path))

        data_dir = os.path.join(test_path, "data/parallel/random_data/")
        ckpt_dir = os.path.join(test_path, "data/parallel/random_ckpt/")
        output_dir = os.path.join(test_path, "data/parallel/output/")

        cmd = f"bash {dir_path}/{scripts_name} "+\
                    f"{layout} "+\
                    f"{devices} "+\
                    f"{tensor_parallel} "+\
                    f"{data_dir} "+\
                    f"{ckpt_dir} "+\
                    f"{output_dir} "+\
                    f"{dir_path}/{log_dir}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.level0
    @pytest.mark.run(order=3)
    def test_compare_parallel(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between torch loss and mindspore loss
        Expectation: no error
        """
        print("os.environ['PYTHONPATH']:", os.environ['PYTHONPATH'])
        print("sys.path:", sys.path)
        print("os.environ:", os.environ)
        dir_path = os.path.split(os.path.realpath(__file__))[0]
        tests_path = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))
        test_input = "/home/workspace/mindspore_dataset/msadapter/test_input"
        test_path = os.path.join(test_input, os.path.relpath(dir_path, tests_path))
        data_dir = os.path.join(test_path, "data/parallel/output")
        compare_types = ["_forward", "_backward"]

        compare_all_data(data_dir, compare_types=compare_types, atol=0.0, rtol=0.0, print_error_point=False, weight_dict=None)