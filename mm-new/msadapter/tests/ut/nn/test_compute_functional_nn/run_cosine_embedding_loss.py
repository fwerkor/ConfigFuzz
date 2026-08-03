import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse


def test_api_completeness():
    x1 = torch.randn(3, 5, dtype=torch.bfloat16)
    x2 = torch.randn(3, 5, dtype=torch.bfloat16)
    target = torch.ones(3)
    y = F.cosine_embedding_loss(x1, x2, target, margin=0, reduction='mean')

    x1 = torch.randn(3, 5, dtype=torch.float32)
    x2 = torch.randn(3, 5, dtype=torch.float32)
    target = torch.ones(3)
    y = F.cosine_embedding_loss(x1, x2, target, margin=1, reduction='sum')

    x1 = torch.randn(3, 5, dtype=torch.float32)
    x2 = torch.randn(3, 5, dtype=torch.float32)
    target = torch.ones(3)
    y = F.cosine_embedding_loss(x1, x2, target, margin=-1, reduction='none')
    assert y.shape == torch.Size([3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
