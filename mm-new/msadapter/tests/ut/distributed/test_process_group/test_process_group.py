import os
import pytest
import time

script_list = ('run_process_group_gloo.py', 'run_vllm_process_group_gloo.py')


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
def test_process_group():
    log_file = "test_process_group.log"
    for script in script_list:
        cmd = (
            f"python {script} "
            f"--test_mode=completeness "
            f"> {log_file} 2>&1"
        )
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        if ret != 0:
            os.system(f"cat {log_file}")
            raise RuntimeError(f"run test script failed, please check log")
        time.sleep(1)
