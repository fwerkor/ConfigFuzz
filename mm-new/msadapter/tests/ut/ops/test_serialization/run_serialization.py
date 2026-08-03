import os
import argparse
import unittest
import torch


class TestSerialization(unittest.TestCase):
    def test(self, mode):
        self.mode = mode

        x = torch.arange(12).reshape(3, 4)
        self.handle(x, '2d.pt')
        self.handle(x.t(), '2d_t.pt')
        self.handle(x.transpose(0, 1), '2d_transpose.pt')
        self.handle(x.to(torch.bfloat16).t(), '2d_t_bf16.pt')

        x = torch.arange(24).reshape(2, 3, 4)
        self.handle(x, '3d.pt')
        self.handle(x.transpose(0, 1), '3d_01.pt')
        self.handle(x.transpose(1, 2), '3d_12.pt')
        self.handle(x.to(torch.bfloat16).transpose(1, 2), '3d_12_bf16.pt')

        x = torch.arange(12).reshape(2, 6)
        self.handle(x[:1], '2d_row_slice1.pt')
        self.handle(x[1:], '2d_row_slice2.pt')
        self.handle(x[:, 0:3], '2d_col_slice1.pt')
        self.handle(x[:, 3:6], '2d_col_slice2.pt')
        self.handle(x.to(torch.bfloat16)[:, 3:6], '2d_col_slice2_bf16.pt')

        x = torch.arange(900).reshape(5, 6, 10, 3)
        self.handle(x, '4d.pt')
        self.handle(x[0:2], '4d1.pt')
        self.handle(x[2:4], '4d2.pt')
        self.handle(x[3:5], '4d3.pt')
        self.handle(x[:, 0:2], '4d4.pt')
        self.handle(x[:, 4:6], '4d5.pt')
        self.handle(x[:, :, 0:3], '4d6.pt')
        self.handle(x[:, :, 3:6], '4d7.pt')
        self.handle(x[:, :, 6:10], '4d8.pt')
        self.handle(x[:, :, 9:10], '4d9.pt')
        self.handle(x[:, :, :, 0:1], '4d10.pt')
        self.handle(x[:, :, :, 1:3], '4d11.pt')
        self.handle(x[:, 1:3, 3:5], '4d12.pt')
        self.handle(x[1:3:, 1:3, 3:5, 1:3], '4d13.pt')
        self.handle(x.to(torch.bfloat16)[1:3:, 1:3, 3:5, 1:3], '4d14.pt')

    def handle(self, tensor, pt_file):
        if self.mode == 1:
            torch.save(tensor, pt_file)
        elif self.mode == 2:
            loaded = torch.load(pt_file, weights_only=True)
            assert torch.equal(tensor, loaded)
            os.remove(pt_file)
        elif self.mode == 3:
            result = torch._utils._is_contiguous_by_shape_stride(tensor.shape, tensor.stride())
            assert result == tensor.is_contiguous()
        else:
            assert False

if __name__ == "__main__":
    print(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")
    print(f"torch: {torch}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['save', 'load', 'is_contiguous'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    print(f"args.test_mode: {args.test_mode}")

    if args.test_mode == 'save':
        TestSerialization().test(1)
    elif args.test_mode == 'load':
        TestSerialization().test(2)
    elif args.test_mode == 'is_contiguous':
        TestSerialization().test(3)
    else:
        assert False
