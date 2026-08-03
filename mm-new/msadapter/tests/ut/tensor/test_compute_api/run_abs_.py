import os
import time
import numpy as np
import torch
import argparse


def test_api_completeness():
    x = torch.randn(2, 3, dtype=torch.bfloat16)
    y = x.abs_()

    x = torch.randn(3, 2, 3, dtype=torch.float32)
    y = x.abs_()
    assert np.allclose(x, y, 0.0, 0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
