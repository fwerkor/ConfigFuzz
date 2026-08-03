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


class TestSerialization:
    script_name = "run_serialization.py"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.level0
    def test_api_completeness(self):
        # torch save
        log_file = "test_torch_save.log"
        cmd = f"python {self.script_name} --test_mode=save > {log_file} 2>&1"
        print(f"\nrun cmd1 is:\n{cmd}")
        ret = os.system(cmd)
        if ret != 0:
            os.system(f"cat {log_file}")
            raise RuntimeError(f"run test script failed(1), please check log")

        init_env('mindspore')

        # msa load
        log_file = "test_msa_load1.log"
        cmd = f"python {self.script_name} --test_mode=load > {log_file} 2>&1"
        print(f"\nrun cmd2 is:\n{cmd}")
        ret = os.system(cmd)
        if ret != 0:
            os.system(f"cat {log_file}")
            raise RuntimeError(f"run test script failed(2), please check log")

        # msa save
        log_file = "test_msa_save.log"
        cmd = f"python {self.script_name} --test_mode=save > {log_file} 2>&1"
        print(f"\nrun cmd3 is:\n{cmd}")
        ret = os.system(cmd)
        if ret != 0:
            os.system(f"cat {log_file}")
            raise RuntimeError(f"run test script failed(3), please check log")

        # msa load
        log_file = "test_msa_load2.log"
        cmd = f"python {self.script_name} --test_mode=load > {log_file} 2>&1"
        print(f"\nrun cmd4 is:\n{cmd}")
        ret = os.system(cmd)
        if ret != 0:
            os.system(f"cat {log_file}")
            raise RuntimeError(f"run test script failed(4), please check log")

        # test is_contiguous
        log_file = "test_is_contiguous.log"
        cmd = f"python {self.script_name} --test_mode=is_contiguous > {log_file} 2>&1"
        print(f"\nrun cmd5 is:\n{cmd}")
        ret = os.system(cmd)
        if ret != 0:
            os.system(f"cat {log_file}")
            raise RuntimeError(f"run test script failed(5), please check log")
