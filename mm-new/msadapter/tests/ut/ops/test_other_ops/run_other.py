import os
import argparse

import torch

from torch_npu.testing.testcase import TestCase

from torch.testing._internal.common_utils import seed_all


class TestCumsum(TestCase):

    def test_api_completeness(self):
        x = torch.randn(10)
        y = torch.cumsum(x, 0)

        x = torch.randn(2, 3)
        y = torch.cumsum(x, 1, dtype=torch.float32)

        x = torch.tensor([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]])
        y = torch.cumsum(x, 1, dtype=torch.bfloat16)

        print("cumsum: ", y)
        benchmark = torch.tensor([[3., 7., 13., 23.],
                                  [1., 7., 14., 23.],
                                  [4., 7., 15., 22.],
                                  [1., 4., 11., 20.]], dtype=torch.bfloat16)
        assert torch.equal(y, benchmark)


class TestTril(TestCase):

    def test_api_completeness(self):
        x = torch.randn(3, 3)
        y = torch.tril(x)

        x = torch.arange(16).reshape((4, 4))
        y = torch.tril(x, diagonal=1)
        print("tril: ", y)

        benchmark = torch.tensor([[0, 1, 0, 0],
                                  [4, 5, 6, 0],
                                  [8, 9, 10, 11],
                                  [12, 13, 14, 15]])
        assert torch.equal(y, benchmark)


class TestHistc(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([1., 2, 1], dtype=torch.float32)
        bins = 4
        y = torch.histc(x, bins=bins, min=0, max=3)
        print("histc: ", y)

        benchmark = torch.tensor([0., 2., 1., 0.], dtype=torch.float32)
        assert torch.equal(y, benchmark)


class TestTriu(TestCase):

    def test_api_completeness(self):
        x = torch.randn(3, 3)
        y = torch.triu(x)

        x = torch.arange(16).reshape((4, 4))
        y = torch.triu(x, diagonal=1)
        print("triu: ", y)

        benchmark = torch.tensor([[0, 1, 2, 3],
                                  [0, 0, 6, 7],
                                  [0, 0, 0, 11],
                                  [0, 0, 0, 0]])
        assert torch.equal(y, benchmark)


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestCumsum().test_api_completeness()
        TestTril().test_api_completeness()
        TestHistc().test_api_completeness()
        TestTriu().test_api_completeness()
