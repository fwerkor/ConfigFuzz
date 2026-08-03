import os
import time
import numpy as np
import torch
import torch.nn as nn
import argparse


def test_api_completeness():
    x = torch.randn(2, dtype=torch.bfloat16)
    m = nn.ReLU()
    y = m(x)

    x = torch.randn(2, 4, dtype=torch.float32)
    m = nn.ReLU(inplace=True)
    y = m(x)
    assert x.shape == torch.Size([2, 4])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
