import os
import time
import numpy as np
import torch
import argparse


def test_api_completeness():
    torch.iinfo(int)
    torch.iinfo(torch.int8)
    torch.iinfo(torch.int16)
    torch.iinfo(torch.int32)
    torch.iinfo(torch.int64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
