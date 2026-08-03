import os
import time
import numpy as np
import torch
import torch.nn as nn
import argparse


def test_api_completeness():
    hidden_states = torch.randn(128, 256)

    dropout = nn.Dropout(p=0.5, inplace=False)
    hidden_states = dropout(input=hidden_states)

    dropout = nn.Dropout()
    hidden_states = dropout(input=hidden_states)

    dropout = nn.Dropout(inplace=True)
    dropout(input=hidden_states)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()

