import os
import time
import numpy as np
import torch
import argparse


def test_api_completeness():
    x = torch.tensor([[0., 0., 0., 0, 0], [0., 0., 0., 0, 0]], dtype=torch.float32)
    mask = torch.tensor([[0., 0., 0., 1, 1], [1., 1., 0., 1, 1]], dtype=torch.bool)
    source = torch.tensor([[0., 1., 2., 3, 4], [5., 6., 7., 8, 9]], dtype=torch.float32)
    y = x.masked_scatter_(mask, source)

    x = torch.tensor([[1., 2., 3., 4, 5], [6., 7., 8., 9, 10]], dtype=torch.float32)
    mask = torch.tensor([[0., 1., 1., 1, 1], [1., 1., 0., 0, 1]], dtype=torch.bool)
    source = torch.tensor([[0., 1., 2., 3, 4], [5., 6., 7., 8, 9]], dtype=torch.float32)
    y = x.masked_scatter_(mask, source)
    assert np.allclose(x, y, 0.0, 0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
