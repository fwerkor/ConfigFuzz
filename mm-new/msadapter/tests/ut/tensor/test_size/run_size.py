import os
import argparse
import unittest
import pytest
import torch


class TestSize(unittest.TestCase):
    def test_api_completeness(self):
        x = torch.ones(10, 20, 30)
        assert x.size(0) == 10
        assert x.size(1) == 20
        assert x.size(2) == 30

        s = x.size()
        assert s == x.shape
        assert s == torch.Size([10, 20, 30])

        assert isinstance(s, torch.Size)
        assert type(s) == torch.Size
        assert len(s) == 3
        assert s.numel() == 6000
        assert s.numel() == x.numel()
        assert s.__reduce__() == (torch.Size, ((10, 20, 30),))

        assert s + s == torch.Size([10, 20, 30, 10, 20, 30])
        assert s * 2 == torch.Size([10, 20, 30, 10, 20, 30])
        assert 2 * s == torch.Size([10, 20, 30, 10, 20, 30])

        assert type(s[1]) == int
        assert type(s[1:]) == torch.Size
        assert s[1] == 20
        assert s[1:] == torch.Size([20, 30])
        assert s[1:].numel() == 600

        s = torch.Size([1, 2, 3, 1, 1])
        assert s.count(1) == 3
        assert s.index(1) == 0
        assert s.index(1, 1) == 3

        with pytest.raises(TypeError):
            _ = torch.Size((1, 'a'))


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance', 'precision', 'outlier'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestSize().test_api_completeness()