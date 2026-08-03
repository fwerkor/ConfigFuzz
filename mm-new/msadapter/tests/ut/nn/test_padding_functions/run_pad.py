import os
import torch
import torch.nn.functional as F

from torch_npu.testing.testcase import TestCase
from torch.testing._internal.common_utils import seed_all


class TestPad(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()
        self.test_api_completeness_3()
        self.test_api_completeness_4()
        self.test_api_completeness_5()

    def test_api_completeness_1(self):
        mode = "constant"
        t4d = torch.empty(3, 3, 4, 2)
        p1d = (1, 1)
        value = 0
        F.pad(t4d, p1d, mode, value)

    def test_api_completeness_2(self):
        mode = "constant"
        t4d = torch.empty(3, 3, 4, 2)
        p1d = (1, 1, 2, 2)
        value = 0
        F.pad(t4d, p1d, mode, value)

    def test_api_completeness_3(self):
        mode = "constant"
        t4d = torch.empty(3, 3, 4, 2)
        p1d = (0, 1, 2, 1, 3, 3)
        value = 0
        F.pad(t4d, p1d, mode, value)

    def test_api_completeness_4(self):
        mode = "reflect"
        t4d = torch.empty(3, 3, 4, 2)
        p1d = (1, 1, 2, 2)
        value = 0
        F.pad(t4d, p1d, mode, value)

    def test_api_completeness_5(self):
        mode = "replicate"
        t4d = torch.empty(3, 3, 4, 2)
        p1d = (1, 1, 2, 2)
        value = 0
        F.pad(t4d, p1d, mode, value)


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    TestPad().test_api_completeness()
