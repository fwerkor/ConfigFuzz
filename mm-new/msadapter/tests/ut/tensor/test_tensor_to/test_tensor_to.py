import os
import pytest
import time

script_list = ('run_tensor_to.py',)


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
def test_tensor_to():
    log_file = "test_tensor_to.log"
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