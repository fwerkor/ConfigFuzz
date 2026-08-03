import os
import time
import numpy as np
import torch
import torch.nn as nn
import argparse

mode = 'mindspore'


def test_api_completeness():
    x = torch.tensor(torch.randn(20, 100, 35, 45, 10), dtype=torch.bfloat16)
    m = nn.BatchNorm3d(100, affine=True, track_running_stats=True, dtype=torch.bfloat16)
    y = m(x)

    x = torch.tensor(torch.randn(20, 100, 35, 45, 10), dtype=torch.float32)
    m = nn.BatchNorm3d(100, affine=True, track_running_stats=False, dtype=torch.float32)
    y = m(x)
    assert y.shape == torch.Size([20, 100, 35, 45, 10])


def test_performance():
    repeat_times = 1000

    benchmark_data = {
        'forward': 0.628,
    }

    forward_cost_time = 0
    backward_cost_time = 0
    x = torch.tensor(torch.randn(20, 100, 35, 45, 10), dtype=torch.bfloat16, requires_grad=True).npu()
    m = nn.BatchNorm3d(100, dtype=torch.bfloat16).npu()

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
    assert res_forward_cost <= benchmark_data['forward'] * 1.2, 'nn.BatchNorm3d, forward performance degradation ' \
                                                                'exceeding 20%'


def test_precision():

    input_path = '/home/workspace/mindspore_dataset/msadapter/test_input/ut'
    data_name = 'BatchNorm3d.pt'

    m = nn.BatchNorm3d(100, dtype=torch.bfloat16).npu()
    x = torch.ones(20, 100, 35, 45, 10,  requires_grad=True, dtype=torch.bfloat16).npu()

    if mode == 'mindspore':
        # forward
        pt_data = torch.load(os.path.join(input_path, data_name))
        pt_y, pt_x_grad = pt_data['pt_y'], pt_data['pt_x_grad']
        ms_y = m(x)
        y_sum = ms_y.sum()
        x.retain_grad()
        y_sum.backward()

        # check forward precision
        ms_y_np = ms_y.to(torch.float32).detach().numpy()
        pt_y_np = pt_y.to(torch.float32).detach().numpy()
        assert np.allclose(ms_y_np, pt_y_np, 0.0, 0.0)

        ms_grad_np = x.grad.to(torch.float32).detach().numpy()
        pt_grad_np = pt_x_grad.to(torch.float32).detach().numpy()
        assert np.allclose(ms_grad_np, pt_grad_np, 0.0, 0.0)

    elif mode == 'torch':
        # forward
        pt_y = m(x)
        y_sum = pt_y.sum()
        x.retain_grad()
        y_sum.backward()

        save_dict = {
            'pt_y': pt_y,
            'pt_x_grad': x.grad,
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
