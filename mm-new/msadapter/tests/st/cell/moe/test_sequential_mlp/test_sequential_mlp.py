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
"""Test SquentialMLP"""
import os
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))   # utils path
from utils_st import compare_all_data, init_env, compare_all_data_2


class TestSquentialMLP:


    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.level1
    @pytest.mark.run(order=2)
    def test_mindspore_squential_mlp(self):
        """
        Feature: test mindspore.
        Description: run mindspore to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_mindspore.sh"
        layout = "SBH"
        devices = "0,1,2,3,4,5,6,7"
        tensor_parallel = 1
        log_dir = "msrun_log"

        init_env(mode='mindspore')
        print("os.environ['PYTHONPATH']:", os.environ['PYTHONPATH'])
        dir_path = os.path.split(os.path.realpath(__file__))[0]
        tests_path = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))
        test_input = "/home/workspace/mindspore_dataset/msadapter/test_input"
        test_path = os.path.join(test_input, os.path.relpath(dir_path, tests_path))

        # test_path = os.path.split(os.path.realpath(__file__))[0]
        data_dir = os.path.join(test_path, "data/random_data/")
        ckpt_dir = os.path.join(test_path, "data/random_ckpt/")

        output_dir = os.path.join(dir_path, "data/output/")

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
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_0.log"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.level1
    @pytest.mark.run(order=3)
    def test_compare(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between torch loss and mindspore loss
        Expectation: no error
        """

        dir_path = os.path.split(os.path.realpath(__file__))[0]
        tests_path = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))
        test_input = "/home/workspace/mindspore_dataset/msadapter/test_input"
        test_path = os.path.join(test_input, os.path.relpath(dir_path, tests_path))

        # test_path = os.path.split(os.path.realpath(__file__))[0]
        data_dir_pt = os.path.join(test_path, "data/output")
        compare_types = ["_forward", "_backward"]

        data_dir_ms = os.path.join(dir_path, "data/output")

        compare_all_data_2(data_dir_pt, data_dir_ms, compare_types=compare_types, atol=0.0, rtol=0.0, print_error_point=False, weight_dict=None)
