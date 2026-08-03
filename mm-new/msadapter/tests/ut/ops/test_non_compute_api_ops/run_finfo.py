import os
import time
import numpy as np
import torch
import argparse


def test_api_completeness():
    torch.finfo(float)
    torch.finfo(torch.float8_e4m3fn)
    torch.finfo(torch.bfloat16)
    torch.finfo(torch.float32)
    torch.finfo(torch.float64)
    torch.finfo(torch.complex64)
    torch.finfo(torch.complex128)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
