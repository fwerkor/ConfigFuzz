import os
import argparse
import torch

from torch_npu.testing.testcase import TestCase

from torch.testing._internal.common_utils import seed_all


class TestGt(TestCase):

    def test_api_completeness(self):
        self.test_gt_when_given_valid_input_then_return_expected_value()
        self.test_gt_when_given_valid_input_then_return_out_param_value()

    def test_gt_when_given_valid_input_then_return_expected_value(self):
        x1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
        x2 = torch.tensor([[1, 1], [4, 4]], dtype=torch.float32)
        y = torch.gt(x1, x2)

        benchmark = torch.tensor([[False, True], [False, False]])
        print("gt: ", y)
        assert torch.equal(y, benchmark)

    def test_gt_when_given_valid_input_then_return_out_param_value(self):
        x1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
        x2 = torch.tensor([[1, 1], [4, 4]], dtype=torch.float32)
        out = torch.empty((2, 2), dtype=torch.bool)
        torch.gt(x1, x2, out=out)
        benchmark = torch.tensor([[False, True], [False, False]])
        assert torch.equal(out, benchmark)


class TestIsclose(TestCase):

    def test_api_completeness(self):
        x1 = torch.randn((2, 3), dtype=torch.bfloat16)
        x2 = torch.randn((2, 3), dtype=torch.bfloat16)
        rtol = 1e-3
        atol = 1e-6
        y = torch.isclose(x1, x2, rtol, atol, equal_nan=True)

        x1 = torch.randn((2, 3), dtype=torch.float16)
        x2 = torch.randn((2, 3), dtype=torch.float16)
        y = torch.isclose(x1, x2)

        benchmark = torch.tensor([[False, False, False], [False, False, False]])
        print("isclose: ", y)
        assert torch.equal(y, benchmark)


class TestSort(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([[8, 2, 1], [5, 9, 3], [4, 6, 7]], dtype=torch.float32)
        y1, y2 = torch.sort(x)

        x = torch.tensor([[8, 2, 1], [5, 9, 3], [4, 6, 7]], dtype=torch.bfloat16)
        dim = 0
        descending = True
        stable = True
        y1, y2 = torch.sort(x, dim=dim, descending=descending, stable=stable)
        print("sort: ", y1, y2)

        benchmark = torch.tensor([[8., 9., 7.], [5., 6., 3., ], [4., 2., 1., ]], dtype=torch.bfloat16)
        assert torch.equal(y1, benchmark)


class TestTopk(TestCase):

    def test_api_completeness(self):
        x = torch.randn((3, 4), dtype=torch.float32)
        k = 2
        dim = 0
        y = torch.topk(x, k, dim=dim)

        x = torch.arange(1., 7., dtype=torch.bfloat16).reshape((2, 3))
        k = 3
        dim = 1
        largest = False
        y = torch.topk(x, k, dim=dim, largest=largest, sorted=False)
        print("topk: ", y)

        benchmark = torch.tensor([[1., 2., 3.], [4., 5., 6., ]], dtype=torch.bfloat16)
        assert torch.equal(y[0], benchmark)


class TestEqual(TestCase):

    def test_api_completeness(self):
        x1 = torch.tensor([1, 2], dtype=torch.bfloat16)
        x2 = torch.tensor([1, 2], dtype=torch.float16)
        y = torch.equal(x1, x2)
        print("equal: ", y)
        assert y == True


class TestAllclose(TestCase):

    def test_api_completeness(self):
        x1 = torch.tensor([1.0, float('nan')], dtype=torch.float32)
        x2 = torch.tensor([1.0, float('nan')], dtype=torch.float32)
        equal_nan = True
        y = torch.allclose(x1, x2, equal_nan=equal_nan)

        x1 = torch.tensor([10000., 1e-07], dtype=torch.float16)
        x2 = torch.tensor([10000.1, 1e-08], dtype=torch.float16)
        rtol = 1e-3
        atol = 1e-5
        equal_nan = False
        y = torch.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        print("allclose: ", y)
        assert y == False


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestGt().test_api_completeness()
        TestIsclose().test_api_completeness()
        TestSort().test_api_completeness()
        TestTopk().test_api_completeness()
        TestEqual().test_api_completeness()
        # TestAllclose().test_api_completeness()
