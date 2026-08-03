import os
import time
import numpy as np
import torch
import torch.nn as nn
import argparse


def test_api_completeness():
    x = torch.randn(2, 1, 3, 3, dtype=torch.float32)
    m = nn.ReplicationPad2d(2)
    y = m(x)
    assert y.shape == torch.Size([2, 1, 7, 7])

    x = torch.randn(2, 1, 3, 3, dtype=torch.float32)
    m = nn.ReplicationPad2d((1, 1, 2, 0))
    y = m(x)
    assert y.shape == torch.Size([2, 1, 5, 5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
