import os
import argparse
import torch

from torch_npu.testing.testcase import TestCase
from torch_npu import npu_matmul_add_fp32, npu_groupmatmul_add_fp32

from torch.testing._internal.common_utils import seed_all


class TestMatmulAdd(TestCase):

    def test_api_completeness(self):
        x = torch.rand((64, 128), dtype=torch.float16).npu()
        weight = torch.rand((64, 128), dtype=torch.float16).npu()
        bias = torch.rand((128, 128), dtype=torch.float32).npu()

        product = torch.matmul(x.T, weight)
        res = product + bias

        npu_matmul_add_fp32(weight, x, bias)

        assert res.shape == bias.shape


class TestGroupMatmulAdd(TestCase):
    
    def test_api_completeness(self):
        x = torch.rand((32, 128), dtype=torch.float16).npu()
        grad_out = torch.rand((32, 64), dtype=torch.float16).npu()
        group_list = torch.tensor([15, 32], dtype=torch.int64).npu()
        out = torch.rand((256, 64), dtype=torch.float32).npu()

        npu_groupmatmul_add_fp32(x, grad_out, group_list, out)

        assert out.shape == (256, 64)




if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestMatmulAdd().test_api_completeness()
        TestGroupMatmulAdd().test_api_completeness()
