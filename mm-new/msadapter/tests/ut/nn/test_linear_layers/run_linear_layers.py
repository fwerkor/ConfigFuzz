import os
import torch
import torch.nn as nn

from torch_npu.testing.testcase import TestCase
from torch.testing._internal.common_utils import seed_all


class TestIdentity(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()

    def test_api_completeness_1(self):
        input_ = torch.randn(128, 20)
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        m(input_)

    def test_api_completeness_2(self):
        input_ = torch.randn(128, 20)
        m = nn.Identity()
        m(input_)


class TestLinear(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()
        self.test_api_completeness_3()
        self.test_api_completeness_4()

    def test_api_completeness_1(self):
        input_ = torch.randn(128, 20)
        bias = True
        dtype = None
        m = nn.Linear(20, 30, bias=bias, dtype=dtype)
        m(input_)

    def test_api_completeness_2(self):
        bias = False
        dtype = torch.float32
        input_ = torch.randn(128, 20, dtype=dtype)
        m = nn.Linear(20, 30, bias=bias, dtype=dtype)
        m(input_)

    def test_api_completeness_3(self):
        bias = True
        dtype = torch.float16
        input_ = torch.randn(128, 20, dtype=dtype)
        m = nn.Linear(20, 30, bias=bias, dtype=dtype)
        m(input_)

    def test_api_completeness_4(self):
        bias = True
        dtype = torch.bfloat16
        input_ = torch.randn(128, 20, dtype=dtype)
        m = nn.Linear(20, 30, bias=bias, dtype=dtype)
        m(input_)


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    TestIdentity().test_api_completeness()
    TestLinear().test_api_completeness()
