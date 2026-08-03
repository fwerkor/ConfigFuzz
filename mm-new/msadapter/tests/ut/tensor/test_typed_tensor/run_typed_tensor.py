import os
import argparse
import unittest
import torch


class TestTypedTensor(unittest.TestCase):
    def test_api_completeness(self):
        value = [0.1, 1.1, 2]
        def test_typed_tensor(cls, dtype):
            typed_tensor = cls(value)
            assert type(typed_tensor) == torch.Tensor
            assert isinstance(typed_tensor, torch.Tensor)
            assert isinstance(typed_tensor, cls)
            assert typed_tensor.dtype == dtype
            assert typed_tensor._ms_device.split(':')[0] == self.device_type

            tensor = torch.tensor(value, dtype=dtype)
            tensor = torch._tensor._ms_to(tensor, self.device_type)
            assert isinstance(tensor, torch.Tensor)
            assert isinstance(tensor, cls)
            assert typed_tensor.nbytes == tensor.nbytes
            assert (typed_tensor.add(1) == tensor.add(1)).all()
            assert (torch.mul(typed_tensor, 2) == torch.mul(tensor, 2)).all()
            assert (typed_tensor * 2 == tensor * 2).all()
            assert (2 * typed_tensor == tensor * 2).all()

            if dtype is not torch.uint8:
                assert not isinstance(typed_tensor, torch.ByteTensor)
            if dtype is not torch.bool:
                assert not isinstance(typed_tensor, torch.BoolTensor)

        def test_module(module, device):
            self.device_type = device

            test_typed_tensor(module.FloatTensor, torch.float32)
            test_typed_tensor(module.DoubleTensor, torch.float64)
            test_typed_tensor(module.HalfTensor, torch.float16)
            test_typed_tensor(module.BFloat16Tensor, torch.bfloat16)

            test_typed_tensor(module.ByteTensor, torch.uint8)
            test_typed_tensor(module.CharTensor, torch.int8)
            test_typed_tensor(module.ShortTensor, torch.int16)
            test_typed_tensor(module.IntTensor, torch.int32)
            test_typed_tensor(module.LongTensor, torch.int64)

            test_typed_tensor(module.BoolTensor, torch.bool)

        test_module(torch, 'CPU')
        test_module(torch.cuda, 'Ascend')

        assert not isinstance(torch.FloatTensor(value), torch.cuda.FloatTensor)
        assert not isinstance(torch.DoubleTensor(value), torch.cuda.DoubleTensor)
        assert not isinstance(torch.HalfTensor(value), torch.cuda.HalfTensor)
        assert not isinstance(torch.BFloat16Tensor(value), torch.cuda.BFloat16Tensor)

        assert not isinstance(torch.cuda.FloatTensor(value), torch.FloatTensor)
        assert not isinstance(torch.cuda.DoubleTensor(value), torch.DoubleTensor)
        assert not isinstance(torch.cuda.HalfTensor(value), torch.HalfTensor)
        assert not isinstance(torch.cuda.BFloat16Tensor(value), torch.BFloat16Tensor)

if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance', 'precision', 'outlier'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestTypedTensor().test_api_completeness()
