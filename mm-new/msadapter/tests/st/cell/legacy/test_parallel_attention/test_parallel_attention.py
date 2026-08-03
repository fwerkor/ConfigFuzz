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
"""Test Parallel_Attention"""
import os
import numpy as np
import pytest
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
from utils_st import compare_all_data, init_env


class TestParallelParallelAttention:
    """A test class for ParallelAttention. """

    test_input = "/home/workspace/mindspore_dataset/msadapter/test_input"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    @pytest.mark.level1
    def test_mindspore_parallel_attention(self):
        """
        Feature: test mindspore parallel_attention.
        Description: run mindspore parallel_attention to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_mindspore.sh"
        layout = "SBH"
        devices = "0,1,2,3,4,5,6,7"
        tensor_parallel = "2"
        log_dir = "msrun_parallel_log"

        init_env(mode='mindspore')
        local_test_path = os.path.split(os.path.realpath(__file__))[0]
        test_path_dir = os.path.dirname(os.path.dirname(os.path.dirname(local_test_path)))
        test_path = os.path.join(self.test_input, os.path.relpath(local_test_path, test_path_dir))

        data_dir = os.path.join(test_path, "data/parallel/random_data/")
        ckpt_dir = os.path.join(test_path, "data/parallel/random_ckpt/")
        output_dir = os.path.join(test_path, "data/parallel/output/")

        cmd = f"bash {local_test_path}/{scripts_name} " + \
              f"{layout} " + \
              f"{devices} " + \
              f"{tensor_parallel} " + \
              f"{data_dir} " + \
              f"{ckpt_dir} " + \
              f"{output_dir} " + \
              f"{local_test_path}/{log_dir}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        assert ret == 0, f"msrun failed, please check {local_test_path}/{log_dir}/worker_*.log"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    @pytest.mark.level1
    def test_compare_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between torch loss and mindspore loss
        Expectation: no error
        """

        local_test_path = os.path.split(os.path.realpath(__file__))[0]
        test_path_dir = os.path.dirname(os.path.dirname(os.path.dirname(local_test_path)))
        test_path = os.path.join(self.test_input, os.path.relpath(local_test_path, test_path_dir))

        data_dir = os.path.join(test_path, "data/parallel/output")

        weight_dict = None
        compare_types = ["_forward", "_backward"]
        compare_all_data(data_dir, compare_types=compare_types, atol=0.0, rtol=0.0, print_error_point=False,
                         weight_dict=weight_dict)
