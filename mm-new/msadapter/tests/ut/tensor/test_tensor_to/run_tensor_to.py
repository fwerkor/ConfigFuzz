import os
import argparse
import unittest

import torch
from torch import Tensor
from torch.configs import MS271

class TestTensorTo(unittest.TestCase):
    def test_api_completeness(self):
        dtypes = [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bfloat16, torch.float16, torch.float32, torch.float64]
        devices = {
            'npu': 'Ascend',
            'npu:1': 'Ascend',
            torch.device('npu'): 'Ascend',
            torch.device('npu:1'): 'Ascend',
            'cuda': 'Ascend',
            'Ascend': 'Ascend',
            'cpu': 'CPU'
        }

        # test device
        for torch_device, ms_device in devices.items():
            x = Tensor(1.0, torch.float32)
            y = x.to(torch_device)
            self.cmp_device(y, ms_device)

        # test dtype
        for dtype in dtypes:
            x = Tensor(1.0, torch.float32)
            y = x.to(dtype)
            assert y.dtype == dtype

        # test device and dtype
        for torch_device, ms_device in devices.items():
            for dtype in dtypes:
                x = Tensor(1.0, torch.float32)
                y = x.to(torch_device, dtype)
                self.cmp_device(y, ms_device)
                assert y.dtype == dtype

        # test kwargs
        for torch_device, ms_device in devices.items():
            for dtype in dtypes:
                for non_blocking in [True, False]:
                    for copy in [True, False]:
                        x = Tensor(1.0, torch.float32)
                        y = x.to(device=torch_device, dtype=dtype, non_blocking=non_blocking, copy=copy)
                        self.cmp_device(y, ms_device)
                        assert dtype == y.dtype
                        if copy:
                            self.cmp_copy(y, x)

        # test args
        for torch_device, ms_device in devices.items():
            for dtype in dtypes:
                for non_blocking in [True, False]:
                    for copy in [True, False]:
                        x = Tensor(1.0, torch.float32)
                        y = x.to(torch_device, dtype, non_blocking, copy)
                        self.cmp_device(y, ms_device)
                        assert dtype == y.dtype
                        if copy:
                            self.cmp_copy(y, x)

        # test to(other, ...)
        for torch_device, ms_device in devices.items():
            for dtype in dtypes:
                for non_blocking in [True, False]:
                    for copy in [True, False]:
                        x = Tensor(1.0, torch.float32)
                        other = Tensor(2.0, dtype).to(torch_device)
                        y = x.to(other, non_blocking, copy)
                        self.cmp_device(y, ms_device)
                        assert dtype == y.dtype
                        if copy:
                            self.cmp_copy(y, x)

        x = Tensor(1.0, torch.float32)

        y = x.to(torch.float64, False, True)
        assert y.dtype == torch.float64
        self.cmp_copy(y, x)

        y = x.to(torch.float64, False, copy=False)
        assert y.dtype == torch.float64

        y = x.to(torch.float64, non_blocking=False, copy=False)
        assert y.dtype == torch.float64

        y = x.to(dtype=torch.float64, non_blocking=False, copy=False)
        assert y.dtype == torch.float64

        y = x.to('npu', torch.float64, False, True)
        self.cmp_device(y, 'Ascend')
        assert y.dtype == torch.float64
        self.cmp_copy(y, x)

        y = x.to('npu', torch.float64, False, copy=False)
        self.cmp_device(y, 'Ascend')
        assert y.dtype == torch.float64

        y = x.to('npu', torch.float64, non_blocking=False, copy=False)
        self.cmp_device(y, 'Ascend')
        assert y.dtype == torch.float64

        y = x.to('npu', dtype=torch.float64, non_blocking=False, copy=False)
        self.cmp_device(y, 'Ascend')
        assert y.dtype == torch.float64

        y = x.to(device='npu', dtype=torch.float64, non_blocking=False, copy=False)
        self.cmp_device(y, 'Ascend')
        assert y.dtype == torch.float64

        other = Tensor(2.0, torch.float64).to('npu')

        y = x.to(other, False, False)
        self.cmp_device(y, other._ms_device)
        assert y.dtype == other.dtype

        y = x.to(other, False, copy=False)
        self.cmp_device(y, other._ms_device)
        assert y.dtype == other.dtype

        y = x.to(other, non_blocking=False, copy=False)
        self.cmp_device(y, other._ms_device)
        assert y.dtype == other.dtype

        y = x.to(other=other, non_blocking=False, copy=False)
        self.cmp_device(y, other._ms_device)
        assert y.dtype == other.dtype

    def cmp_device(self, actual, expected):
        if MS271:
            assert expected in actual._ms_device

    def cmp_copy(self, actual, expected):
        if MS271:
            assert actual.data_ptr() != expected.data_ptr()

if __name__ == "__main__":
    print(f"PYTHONPATH is: {os.getenv('PYTHONPATH')}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance', 'precision', 'outlier'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestTensorTo().test_api_completeness()
