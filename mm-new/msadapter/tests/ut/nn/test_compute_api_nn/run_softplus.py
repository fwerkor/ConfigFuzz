import os
import time
import numpy as np
import torch
import torch.nn as nn
import argparse

mode = 'mindspore'


def test_api_completeness():
    x = torch.tensor(torch.randn(1, 4, 8), dtype=torch.bfloat16)
    m = nn.Softplus()
    y = m(x)

    x = torch.tensor(torch.randn(1, 4, 8), dtype=torch.float32)
    m = nn.Softplus(2, 30)
    y = m(x)
    assert y.shape == torch.Size([1, 4, 8])


def test_performance():
    repeat_times = 1000

    benchmark_data = {
        'forward': 0.1115,
        'backward': 0.3968
    }

    forward_cost_time = 0
    backward_cost_time = 0
    x = torch.tensor(torch.randn(1, 4, 8), requires_grad=True, dtype=torch.bfloat16).npu()
    m = nn.Softplus().npu()

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

    assert res_forward_cost <= benchmark_data['forward'] * 1.2, 'nn.Softplus, forward performance degradation ' \
                                                                'exceeding 20%'
    assert res_backward_cost <= benchmark_data['backward'] * 1.2, 'nn.Softplus, backward performance degradation ' \
                                                                  'exceeding 20%'


def test_precision():
    input_path = '/home/workspace/mindspore_dataset/msadapter/test_input/ut'
    data_name = 'Softplus.pt'

    m = nn.Softplus().npu()

    if mode == 'mindspore':
        pt_data = torch.load(os.path.join(input_path, data_name))
        pt_x, pt_y, pt_x_grad = pt_data['pt_x'], pt_data['pt_y'], pt_data['pt_x_grad']
        pt_x.requires_grad = True

        ms_y = m(pt_x)
        ms_y_sum = ms_y.sum()
        ms_y_sum.backward()

        # check forward precision
        ms_y_np = ms_y.to(torch.float32).detach().numpy()
        pt_y_np = pt_y.to(torch.float32).detach().numpy()
        assert np.allclose(ms_y_np, pt_y_np, 0.0, 0.0)

        # check backward precision
        assert pt_x.grad is not None, f'Expect grad_list is not None, bug got {pt_x.grad}'
        ms_grad_np = pt_x.grad.to(torch.float32).detach().numpy()
        pt_grad_np = pt_x_grad.to(torch.float32).detach().numpy()
        assert np.allclose(ms_grad_np, pt_grad_np, 0.0, 0.0)

    elif mode == 'torch':
        # forward
        x = torch.tensor(torch.randn(1, 4, 8), requires_grad=True, dtype=torch.bfloat16)
        y = m(x.to('npu'))
        y_sum = y.sum()
        y_sum.backward()

        save_dict = {
            'pt_x': x,
            'pt_y': y,
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
