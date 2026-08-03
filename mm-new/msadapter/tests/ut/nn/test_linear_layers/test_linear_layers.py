import os
import pytest
from pathlib import Path


def init_env(mode='megatron'):
    msadapter_path = Path.cwd().parents[3] / 'msadapter'
    msa_thirdparty_path = msadapter_path / 'msa_thirdparty'
    python_path_ori = os.getenv('PYTHONPATH').replace(msadapter_path.__str__(), '').replace(
        msa_thirdparty_path.__str__(), '')
    python_path_msadapter = f"{msadapter_path}:{msa_thirdparty_path}:{python_path_ori}"

    if mode == 'megatron':
        os.environ['PYTHONPATH'] = python_path_ori
    elif mode == 'mindspore':
        os.environ['PYTHONPATH'] = python_path_msadapter
    print('## PYTHONPATH: ', os.getenv('PYTHONPATH'))


class TestLinearLayers:
    script_name = "run_linear_layers.py"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.level0
    @pytest.mark.env_onecard
    def test_api_completeness(self):
        init_env('mindspore')
        log_file = 'test_linear_layers_completeness.log'
        cmd = (
            f'python {self.script_name}'
            f' > {log_file} 2>&1'
        )
        print(f'\nrun cmd is:\n{cmd}')
        ret = os.system(cmd)
        assert ret == 0, f'run test script failed, please check {log_file}'

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_performance_ms(self):
        init_env('mindspore')
        log_file = 'test_linear_functions_performance_ms.log'
        cmd = (
            f'python {self.script_name} '
            f'--test_mode=performance '
            f'--backend=mindspore '
            f' > {log_file} 2>&1'
        )
        print(f'\nrun cmd is:\n{cmd}')
        ret = os.system(cmd)
        if ret != 0:
            os.system(f"cat {log_file}")
            raise RuntimeError(f"run test script failed, please check log")
