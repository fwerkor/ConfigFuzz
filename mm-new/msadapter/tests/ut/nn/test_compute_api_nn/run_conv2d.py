import os
import numpy as np
import torch
import torch.nn as nn
import argparse

mode = 'mindspore'


def test_api_completeness():
    x = torch.tensor(torch.randn(2, 16, 50, 100), dtype=torch.float32)
    m = nn.Conv2d(16, 33, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros',
                  dtype=torch.float32)
    y = m(x)
    assert y.shape == torch.Size([2, 33, 25, 50])

    x = torch.tensor(torch.randn(2, 16, 50, 100), dtype=torch.bfloat16)
    m = nn.Conv2d(16, 33, kernel_size=(3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), groups=1, bias=False,
                  padding_mode='reflect', dtype=torch.bfloat16)
    y = m(x)
    assert y.shape == torch.Size([2, 33, 26, 100])

    x = torch.tensor(torch.randn(2, 16, 50, 100), dtype=torch.float32)
    m = nn.Conv2d(16, 33, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate',
                  dtype=torch.float32)
    y = m(x)
    assert y.shape == torch.Size([2, 33, 25, 50])


def test_precision():

    input_path = '/home/workspace/mindspore_dataset/msadapter/test_input/ut'
    data_name = 'Conv2d.pt'

    m = nn.Conv2d(16, 33, 3, stride=2, dtype=torch.bfloat16)
    x = torch.ones(2, 16, 50, 100, requires_grad=True, dtype=torch.bfloat16)
    weight = torch.ones(33, 16, 3, 3, requires_grad=True, dtype=torch.bfloat16)
    m.weight = nn.Parameter(weight)
    if mode == 'mindspore':
        pt_data = torch.load(os.path.join(input_path, data_name))
        pt_y, pt_x_grad = pt_data['pt_y'], pt_data['pt_x_grad']
        # forward
        ms_y = m(x)
        ms_y_sum = ms_y.sum()
        ms_y_sum.backward()

        # check forward precision
        ms_y_np = ms_y.to(torch.float32).detach().numpy()
        pt_y_np = pt_y.to(torch.float32).detach().numpy()
        assert np.allclose(ms_y_np, pt_y_np, 0.0, 0.0)

        # check backward precision
        assert x.grad is not None, f'Expect x.grad is not None, bug got {x.grad}'
        ms_grad_np = x.grad.to(torch.float32).detach().numpy()
        pt_grad_np = pt_x_grad.to(torch.float32).detach().numpy()
        assert np.allclose(ms_grad_np, pt_grad_np, 0.0, 0.0)

    elif mode == 'torch':
        # forward
        x = x.to('npu')
        m = m.to('npu')
        y = m(x)
        y_sum = y.sum()
        x.retain_grad()
        y_sum.backward()

        save_dict = {
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
    elif args.test_mode == 'precision':
        test_precision()
