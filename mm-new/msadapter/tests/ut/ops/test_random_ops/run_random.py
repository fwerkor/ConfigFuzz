import os
import time
import random
import argparse
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase
import numpy as np


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


class TestMultinomial(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([[0, 10, 3, 0], [0, 10, 3, 0]], dtype=torch.float32)
        num_samples = 2
        y = torch.multinomial(x, num_samples)

        x = torch.tensor([[0, 10, 3, 0], [0, 10, 3, 0]], dtype=torch.bfloat16)
        num_samples = 3
        g = torch.Generator(device='cuda')
        y = torch.multinomial(x, num_samples, replacement=True, generator=g)

        print("multinomial: ", y)

        benchmark = torch.tensor([[1, 2, 1], [1, 2, 1]])
        assert torch.equal(y, benchmark)


class TestNormal(TestCase):

    def test_api_completeness(self):
        mean = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        std = 1.0
        y = torch.normal(mean, std)

        mean = 1.0
        std = 2.0
        size = (2, 4)
        y = torch.normal(mean, std, size)
        print("normal: ", y)

        benchmark = torch.tensor([[3.4507775, -3.8067923, -0.3221755, 0.21491957],
                                  [-1.6801004, 0.40053993, 0.63404095, -0.05613673]])

        assert torch.allclose(y, benchmark, rtol=1e-3, atol=1e-5)


class TestRandlike(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([[2, 3, 4], [1, 2, 3]], dtype=torch.bfloat16)
        y = torch.rand_like(x)

        x = torch.tensor([[2, 3, 4], [1, 2, 3]], dtype=torch.bfloat16)
        dtype = torch.float32
        y = torch.rand_like(x, dtype=dtype)

        assert y.shape == x.shape


class TestRandn(TestCase):

    def test_api_completeness(self):
        size = 4
        y = torch.randn(size, generator=None, layout=None, device=None, requires_grad=False, pin_memory=False)

        size = (2, 3)
        y = torch.randn(size, dtype=torch.bfloat16)
        assert y.shape == (2, 3)

    def test_performance(self, backend):

        repeat_times = 100

        size = (2, 3)

        # forward
        y = torch.randn(size, dtype=torch.bfloat16, requires_grad=True)

        start_time = time.time()
        for _ in range(repeat_times):
            y = torch.randn(size, dtype=torch.bfloat16, requires_grad=True)
            print(y)

        cost_time = time.time() - start_time
        print(f"repeat {repeat_times} times cost time {cost_time}s")

        # not support for torch
        if backend == "torch":
            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")
            torch.save(cost_time, f".tmp/cost_time.pt")

        elif backend == "mindspore":
            pt_cost_time = 0.15727
            if cost_time > pt_cost_time / 0.3:
                raise ValueError(f"Expect ms cost time <= (pt cost time / 0.3), but got "
                                 f"ms cost time {cost_time}, and pt cost time {pt_cost_time}.")


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    parser.add_argument('--backend', type=str, choices=['torch', 'mindspore'],
                        help="backend")

    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestMultinomial().test_api_completeness()
        TestNormal().test_api_completeness()
        TestRandlike().test_api_completeness()
        TestRandn().test_api_completeness()

    elif args.test_mode == 'performance':
        TestRandn().test_performance(args.backend)
