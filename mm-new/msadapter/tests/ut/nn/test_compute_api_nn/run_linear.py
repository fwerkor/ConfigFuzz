import os
import time
import numpy as np
import torch
import torch.nn as nn
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
from utils_ut import seed_all

mode = 'mindspore'


def test_api_completeness():
    x = torch.randn(128, 20)
    m = nn.Linear(20, 30, bias=True, dtype=None)
    m(x)

    x = torch.randn(128, 20, dtype=torch.float32)
    m = nn.Linear(20, 30, bias=False, dtype=torch.float32)
    m(x)

    x = torch.randn(128, 20, dtype=torch.bfloat16)
    m = nn.Linear(20, 30, bias=True, dtype=torch.bfloat16)
    m(x)


def test_performance():
    repeat_times = 1000

    benchmark_data = {
        'forward': 0.0484,
        'backward': 0.4412
    }

    forward_cost_time = 0
    backward_cost_time = 0
    x = torch.randn(128, 20, requires_grad=True, dtype=torch.float32).npu()
    m = nn.Linear(20, 30, bias=False, dtype=torch.float32)

    for _ in range(repeat_times):
        # forward
        forward_start_time = time.time()
        y = m(x)
        forward_cost_time += (time.time() - forward_start_time)
        y = y.sum()

        # backward
        backward_start_time = time.time()
        y.backward()
        backward_cost_time += (time.time() - backward_start_time)

    res_forward_cost = round(forward_cost_time * 1000 / repeat_times, 4)
    res_backward_cost = round(backward_cost_time * 1000 / repeat_times, 4)

    assert res_forward_cost <= benchmark_data['forward'] * 1.2, 'nn.Linear, forward performance degradation ' \
                                                                'exceeding 20%'
    assert res_backward_cost <= benchmark_data['backward'] * 1.2, 'nn.Linear, backward performance degradation ' \
                                                                  'exceeding 20%'


def test_precision():
    seed_all()

    input_path = '/home/workspace/mindspore_dataset/msadapter/test_input/ut'
    data_name = 'Linear.pt'

    x = torch.ones(128, 20, requires_grad=True, dtype=torch.float32).npu()
    m = nn.Linear(20, 30, bias=False, dtype=torch.float32)
    weight = torch.ones(30, 20, requires_grad=True, dtype=torch.float32).npu()
    m.weight = nn.Parameter(weight)

    if mode == 'mindspore':
        pt_data = torch.load(os.path.join(input_path, data_name))
        pt_y, pt_x_grad = pt_data['pt_y'], pt_data['pt_x_grad']

        ms_y = m(x)
        ms_y_sum = ms_y.sum()
        x.retain_grad()
        ms_y_sum.backward()

        # check forward precision
        assert torch.allclose(ms_y, pt_y, 0.0, 0.0)

        # check backward precision
        assert torch.allclose(x.grad, pt_x_grad, 0.0, 0.0)

    elif mode == 'torch':
        # forward
        pt_y = m(x)
        y_sum = pt_y.sum()
        x.retain_grad()
        y_sum.backward()

        save_dict = {
            'pt_y': pt_y,
            'pt_x_grad': x.grad
        }

        if not os.path.exists(input_path):
            os.makedirs(input_path)
        torch.save(save_dict, os.path.join(input_path, data_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    args, _ = parser.parse_known_args()
    if args.test_mode == 'completeness':
        test_api_completeness()
    elif args.test_mode == 'performance':
        test_performance()
    elif args.test_mode == 'precision':
        test_precision()
