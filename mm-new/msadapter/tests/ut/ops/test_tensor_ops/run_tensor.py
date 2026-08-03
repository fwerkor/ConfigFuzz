import os
import argparse
import unittest
import pytest
import torch


class TestIsTensor(unittest.TestCase):
    def test_api_completeness(self):
        a = torch.tensor([1, 2], dtype=torch.float32)
        assert torch.is_tensor(a)

        a = torch.ByteTensor([1, 2])
        assert torch.is_tensor(a)

        a = torch.BoolTensor([0, 1])
        assert torch.is_tensor(a)

class TestIsFloatingPoint(unittest.TestCase):
    def test_api_completeness(self):
        assert torch.is_floating_point(torch.tensor(1, dtype=torch.bfloat16))
        assert torch.is_floating_point(torch.tensor(1, dtype=torch.float16))
        assert torch.is_floating_point(torch.tensor(1, dtype=torch.float32))
        assert torch.is_floating_point(torch.tensor(1, dtype=torch.float64))
        assert not torch.is_floating_point(torch.tensor(1, dtype=torch.int32))

        with pytest.raises(TypeError):
            torch.is_floating_point(10.)
        with pytest.raises(TypeError):
            torch.is_floating_point(torch.float32)

class TestIsComplex(unittest.TestCase):
    def test_api_completeness(self):
        real = torch.tensor([1, 2], dtype=torch.float32)
        imag = torch.tensor([3, 4], dtype=torch.float32)
        z = torch.complex(real, imag)
        assert z.is_complex()
        assert torch.is_complex(z)
        assert not torch.is_complex(real)
        assert not torch.is_complex(imag)

        real = torch.tensor([1, 2], dtype=torch.float64)
        imag = torch.tensor([3, 4], dtype=torch.float64)
        z = torch.complex(real, imag)
        assert z.is_complex()
        assert torch.is_complex(z)

        with pytest.raises(TypeError):
            c = complex(2, 4)
            torch.is_complex(c)
        with pytest.raises(TypeError):
            torch.is_complex(10.)
        with pytest.raises(TypeError):
            torch.is_complex(torch.complex64)


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance', 'precision', 'outlier'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestIsTensor().test_api_completeness()
        TestIsFloatingPoint().test_api_completeness()
        TestIsComplex().test_api_completeness()
