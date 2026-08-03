import os
import time
import numpy as np
import torch
import argparse


def test_api_completeness():
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.bfloat16)
    y = x.as_strided((2, 2), (1, 2))

    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
    y = x.as_strided((6, 3), (1, 1))
    assert np.allclose(y, torch.tensor([[1., 2., 3.],
                                        [2., 3., 4.],
                                        [3., 4., 5.],
                                        [4., 5., 6.],
                                        [5., 6., 7.],
                                        [6., 7., 8.]], dtype=torch.float32), 0.0, 0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
