import os
import pytest


class TestNonLiAcFunctions:
    script_name = "run_nonlinear_activation_functions.py"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.level0
    def test_api_completeness(self):
        log_file = 'test_nonlinear_activation_functions_completeness.log'
        cmd = (
            f'python {self.script_name}'
            f" --test_mode=completeness "
            f' > {log_file} 2>&1'
        )
        print(f'\nrun cmd is:\n{cmd}')
        ret = os.system(cmd)
        assert ret == 0, f'run test script failed, please check {log_file}'

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.level0
    def test_performance_ms(self):
        log_file = 'test_nonlinear_activation_performance_ms.log'
        cmd = (
            f'python {self.script_name} '
            f'--test_mode=performance '
            f'--backend=mindspore '
            f' > {log_file} 2>&1'
        )
        print(f'\nrun cmd is:\n{cmd}')
        ret = os.system(cmd)
        os.system("cat test_nonlinear_activation_performance_ms.log")  # just for pipeline debug
        assert ret == 0, f'run test script failed, please check {log_file}'
