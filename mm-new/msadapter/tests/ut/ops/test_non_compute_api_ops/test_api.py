import os
import pytest
import time

script_list = ('run_iinfo.py', 'run_finfo.py')


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
def test_api_completeness():
    log_file = "test_api_completeness.log"
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


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level1
def test_api_performance():
    log_file = "test_api_performance.log"
    for script in script_list:
        cmd = (
            f"python {script} "
            f"--test_mode=performance "
            f"> {log_file} 2>&1"
        )
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        if ret != 0:
            os.system(f"cat {log_file}")
            raise RuntimeError(f"run test script failed, please check log")
        time.sleep(1)


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level1
def test_api_precision():
    log_file = "test_api_precision.log"
    for script in script_list:
        cmd = (
            f"python {script} "
            f"--test_mode=precision "
            f"> {log_file} 2>&1"
        )
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        if ret != 0:
            os.system(f"cat {log_file}")
            raise RuntimeError(f"run test script failed, please check log")
        time.sleep(1)
