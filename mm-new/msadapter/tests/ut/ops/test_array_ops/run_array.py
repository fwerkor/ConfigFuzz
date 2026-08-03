import os
import argparse
import torch

from torch_npu.testing.testcase import TestCase

from torch.testing._internal.common_utils import seed_all


class TestCat(TestCase):

    def test_api_completeness(self):
        x = torch.randn(2, 3)
        y = torch.cat((x, x, x), 0)
        print(y)

        x = torch.randn(2, 3)
        y = torch.cat((x, x, x), 1)
        print(y)

        x1 = torch.randn(2, 3)
        x2 = torch.randn(3, 3)
        y = torch.cat((x1, x2), 0)

        assert y.shape == (5, 3)


class TestChunk(TestCase):

    def test_api_completeness(self):
        x = torch.arange(11)
        y = torch.chunk(x, 6)
        print(y)

        x = torch.arange(12).view((3, 4))
        y = torch.chunk(x, 2, 1)
        print(y)

        assert len(y) == 2 and y[0].shape == (3, 2) and y[1].shape == (3, 2)


class TestGather(TestCase):

    def test_api_completeness(self):
        x = torch.randn(3, 4)
        dim = 0
        index = torch.tensor([[0, 0], [1, 1]])
        y = torch.gather(x, dim, index)
        print(y)

        x = torch.randn(3, 4)
        dim = 1
        index = torch.tensor([[0, 0], [1, 1]])
        y = torch.gather(x, dim, index)
        print(y)

        assert y.shape == (2, 2)


class TestNonzero(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                          [0.0, 0.4, 0.0, 0.0],
                          [0.0, 0.0, 1.2, 0.0],
                          [0.0, 0.0, 0.0, -0.4]])
        y = torch.nonzero(x)
        print(y)
        assert torch.equal(y, torch.tensor([[0, 0],
                                            [1, 1],
                                            [2, 2],
                                            [3, 3]]))

        x = torch.tensor([[[0.6, 0.0, 0.0, 0.0],
                           [0.0, 0.4, 0.0, 0.0],
                           [0.0, 0.0, 1.2, 0.0],
                           [0.0, 0.0, 0.0, -0.4]],

                          [[0.6, 0.0, 0.0, 0.0],
                           [0.0, 0.4, 0.0, 0.0],
                           [0.0, 0.0, 1.2, 0.0],
                           [0.0, 0.0, 0.0, -0.4]]])
        y = torch.nonzero(x, as_tuple=True)
        print(y)


class TestScatter(TestCase):

    def test_api_completeness(self):
        dim = 0
        index = torch.tensor([[0, 1, 2, 0]])
        src = torch.arange(1, 11).reshape((2, 5))
        x = torch.zeros(3, 5, dtype=src.dtype)
        y = torch.scatter(x, dim, index, src)
        print(y)

        dim = 1
        index = torch.tensor([[0, 2, 4], [0, 2, 4], [0, 2, 4]])
        src = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.bfloat16)
        x = torch.zeros(5, 5, dtype=src.dtype)
        y = torch.scatter(x, dim, index, src)
        print(y)

        expected_result = torch.tensor([[1., 0., 2., 0., 3.],
                                        [4., 0., 5., 0., 6.],
                                        [7., 0., 8., 0., 9.],
                                        [0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0.]], dtype=torch.bfloat16)
        assert torch.equal(y, expected_result)


class TestSplit(TestCase):

    def test_api_completeness(self):
        x = torch.arange(10).reshape(5, 2)
        y = torch.split(x, 2)
        print(y)

        x = torch.arange(10).reshape(5, 2)
        y = torch.split(x, [1, 4])
        print(y)
        assert len(y) == 2 and y[0].shape == (1, 2) and y[1].shape == (4, 2)


class TestStack(TestCase):

    def test_api_completeness(self):
        x = torch.randn(2, 3)
        y = torch.stack((x, x, x), 0)
        print(y)

        x = torch.randn(2, 3)
        y = torch.stack((x, x, x), 1)
        print(y)
        assert y.shape == (2, 3, 3)


class TestWhere(TestCase):

    def test_api_completeness(self):
        x1 = torch.randn(3, 2)
        x2 = torch.ones(3, 2)
        y = torch.where(x1 > 0, x1, x2)
        print(y)

        x = torch.randn(2, 2, dtype=torch.double)
        y = torch.where(x > 0, x, 0.)
        print(y)

        x = torch.arange(9).reshape(3, 3)
        y = torch.where(x > 4, x, 0)
        expected_result = torch.tensor([[0, 0, 0],
                                        [0, 0, 5],
                                        [6, 7, 8]])
        assert torch.equal(y, expected_result)


class TestConcat(TestCase):

    def test_api_completeness(self):
        x = torch.randn(2, 3)
        y = torch.concat((x, x, x), 0)
        print(y)

        x = torch.randn(2, 3)
        y = torch.concat((x, x, x), 1)
        print(y)

        assert y.shape == (2, 9)


class TestUnbind(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=torch.bfloat16)

        y = torch.unbind(x)
        print(y)
        assert (
                len(y) == 3 and
                torch.equal(y[0], torch.tensor([1, 2, 3], dtype=torch.bfloat16)) and
                torch.equal(y[1], torch.tensor([4, 5, 6], dtype=torch.bfloat16)) and
                torch.equal(y[2], torch.tensor([7, 8, 9], dtype=torch.bfloat16))
        )

        x = torch.tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=torch.float64)

        y = torch.unbind(x, dim=1)
        print(y)


class TestUnsqueeze(TestCase):

    def test_api_completeness(self):
        x = torch.randn(2, 2, dtype=torch.bfloat16)

        y = torch.unsqueeze(x, dim=0)
        print(y)
        assert y.shape == (1, 2, 2)

        x = torch.randn(2, 2, dtype=torch.float64)
        y = torch.unsqueeze(x, dim=1)
        print(y)
        assert y.shape == (2, 1, 2)


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestCat().test_api_completeness()
        TestChunk().test_api_completeness()
        TestGather().test_api_completeness()
        TestNonzero().test_api_completeness()
        TestScatter().test_api_completeness()
        TestSplit().test_api_completeness()
        TestStack().test_api_completeness()
        TestWhere().test_api_completeness()
        TestConcat().test_api_completeness()
        TestUnbind().test_api_completeness()
        TestUnsqueeze().test_api_completeness()
