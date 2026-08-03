import os
import argparse
import torch

from torch_npu.testing.testcase import TestCase
import numpy as np

from torch.testing._internal.common_utils import seed_all


class TestBaddbmm(TestCase):

    def test_api_completeness(self):
        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        y = torch.baddbmm(x, batch1, batch2)
        print(y)

        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        alpha = 2
        beta = 3
        y = torch.baddbmm(x, batch1, batch2, alpha=alpha, beta=beta)
        print(y)

        x = x.numpy()
        batch1 = batch1.numpy()
        batch2 = batch2.numpy()
        y_np = beta * x + alpha * np.matmul(batch1, batch2)
        y_ms = y.numpy()

        assert np.allclose(y_ms, y_np, 1e-3, 1e-3)


class TestOuter(TestCase):

    def test_api_completeness(self):
        x1 = torch.arange(1., 5.)
        x2 = torch.arange(1., 4.)
        y = torch.outer(x1, x2)
        print(y)

        expected_result = torch.tensor([[1., 2., 3.],
                                        [2., 4., 6.],
                                        [3., 6., 9.],
                                        [4., 8., 12.]])

        assert torch.equal(expected_result, y)

        x1 = x1.to(torch.bfloat16)
        x2 = x2.to(torch.bfloat16)
        y = torch.outer(x1, x2)
        print(y)

        x1 = x1.to(torch.float16)
        x2 = x2.to(torch.float16)
        y = torch.outer(x1, x2)
        print(y)


class TestBmm(TestCase):

    def test_api_completeness(self):
        x1 = torch.randn(10, 3, 4)
        x2 = torch.randn(10, 4, 5)

        y = torch.bmm(x1, x2)
        print(y)

        x1 = x1.numpy()
        x2 = x2.numpy()
        y_np = np.matmul(x1, x2)
        y_ms = y.numpy()

        assert np.allclose(y_ms, y_np, 1e-3, 1e-3)

        x1 = torch.randn(10, 3, 4).to(torch.bfloat16)
        x2 = torch.randn(10, 4, 5).to(torch.bfloat16)

        y = torch.bmm(x1, x2)
        print(y)

        x1 = x1.to(torch.float16)
        x2 = x2.to(torch.float16)

        y = torch.bmm(x1, x2)
        print(y)


class TestMatmul(TestCase):

    def test_api_completeness(self):
        x1 = torch.randn(3, dtype=torch.bfloat16)
        x2 = torch.randn(3, dtype=torch.bfloat16)
        y = torch.matmul(x1, x2)
        print(y)

        x1 = torch.randn(10, 3, 4, dtype=torch.float32)
        x2 = torch.randn(4, dtype=torch.float32)
        y = torch.matmul(x1, x2)
        print(y)

        x1 = x1.numpy()
        x2 = x2.numpy()
        y_np = np.matmul(x1, x2)
        y_ms = y.numpy()
        assert np.allclose(y_ms, y_np, 1e-3, 1e-3)


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestBaddbmm().test_api_completeness()
        TestOuter().test_api_completeness()
        TestBmm().test_api_completeness()
        TestMatmul().test_api_completeness()
