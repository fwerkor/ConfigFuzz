import os
import time
import numpy as np
import torch
import argparse


def test_api_completeness():
    x = torch.tensor([1., 2., 3., 4.], dtype=torch.float32)
    end = torch.tensor([10., 10., 10., 10.], dtype=torch.float32)
    weight = 0.5
    y = x.lerp_(end, weight)

    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    end = torch.tensor([10., 20., 30.], dtype=torch.float32)
    weight = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    y = x.lerp_(end, weight)
    assert np.allclose(x, y, 0.0, 0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
