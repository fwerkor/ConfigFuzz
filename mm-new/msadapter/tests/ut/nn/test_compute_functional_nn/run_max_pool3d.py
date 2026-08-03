import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse


def test_api_completeness():
    x = torch.randn(20, 16, 50, 44, 31, dtype=torch.float32)
    y = F.max_pool3d(x, 3, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=False)
    assert y.shape == torch.Size([20, 16, 24, 21, 15])

    x = torch.randn(20, 16, 50, 44, 31, dtype=torch.float32)
    y, _ = F.max_pool3d(x, (3, 2, 2), stride=(2, 1, 2), padding=1, dilation=2, ceil_mode=True, return_indices=True)
    assert y.shape == torch.Size([20, 16, 25, 44, 16])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
