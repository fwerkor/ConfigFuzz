import os
import time
import numpy as np
import torch
import argparse


def test_api_completeness():
    x = torch.randn(3, 3, dtype=torch.float32)
    t1 = torch.randn(3, 1, dtype=torch.float32)
    t2 = torch.randn(1, 3, dtype=torch.float32)
    y = x.addcmul_(t1, t2, value=0.2)
    assert np.allclose(x, y, 0.0, 0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
