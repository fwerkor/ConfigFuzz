import os
import time
import numpy as np
import torch
import torch.nn as nn
import argparse


def test_api_completeness():
    x = torch.tensor(torch.randn(1, 64, 8), dtype=torch.bfloat16)
    m = nn.AdaptiveAvgPool1d(4)
    y = m(x)
    assert y.shape == torch.Size([1, 64, 4])

    x = torch.tensor(torch.randn(4, 8), dtype=torch.float32)
    m = nn.AdaptiveAvgPool1d((2,))
    y = m(x)
    assert y.shape == torch.Size([4, 2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
