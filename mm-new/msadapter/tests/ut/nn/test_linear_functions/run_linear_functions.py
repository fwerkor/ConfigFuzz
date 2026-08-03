import os
import time
import random
import argparse

import torch
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

import numpy as np


class TestLinearFunctions(TestCase):
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    def test_api_completeness(self):
        # pta: torch.nn.functional.linear(input, weight, bias=None)
        # ms: mindspore.mint.nn.functional.linear(input, weight, bias=None)
        inputs = torch.randn(32, 256).to(torch.bfloat16)
        weight = torch.randn(16, 256).to(torch.bfloat16)
        bias = torch.randn(16).to(torch.bfloat16)

        outputs = F.linear(inputs, weight)
        print(outputs)

        outputs = F.linear(input=inputs, weight=weight, bias=None)
        print(outputs)

        outputs = F.linear(inputs, weight, bias)
        print(outputs)

        inputs = inputs.to(torch.float16)
        weight = weight.to(torch.float16)
        bias = bias.to(torch.float16)

        outputs = F.linear(inputs, weight)
        print(outputs)
        outputs = F.linear(inputs, weight, bias)
        print(outputs)

        inputs = inputs.to(torch.float32)
        weight = weight.to(torch.float32)
        bias = bias.to(torch.float32)

        outputs = F.linear(inputs, weight)
        print(outputs)
        outputs = F.linear(inputs, weight, bias)
        print(outputs)

        inputs = inputs.numpy()
        weight = weight.numpy()
        bias = bias.numpy()

        outputs_np = np.dot(inputs, weight.T) + bias
        outputs_ms = outputs.numpy()
        assert np.allclose(outputs_ms, outputs_np, 1e-3, 1e-3)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):
        input_ = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
        weight = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
        bias = torch.randn(shape[0], dtype=torch.bfloat16, requires_grad=True).npu()

        forward_cost_time = 0
        backward_cost_time = 0

        pt_forward_time_diff_shape = {
            (10, 10): 16.3389,
            (100, 100): 6.9274,
            (1000, 100): 4.2620,
            (1000, 1000): 3.3316
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.8732,
            (100, 100): 0.4907,
            (1000, 100): 3.9302,
            (1000, 1000): 5.2741
        }

        if backend == 'torch':
            for _ in range(repeat_times):
                forward_start_time = time.time()
                out = F.linear(input_, weight, bias)
                print(out)
                forward_cost_time += (time.time() - forward_start_time)

                out = out.sum()
                backward_start_time = time.time()
                out.backward()
                backward_cost_time += (time.time() - backward_start_time)
            print(f'shape {shape} single forward cost time {forward_cost_time * 1000 / repeat_times} ms, '
                  f'single backward cost time {backward_cost_time * 1000 / repeat_times} ms')

            if not os.path.exists('.tmp'):
                os.mkdir('.tmp')
            torch.save({
                'forward_cost_time': forward_cost_time,
                'backward_cost_time': backward_cost_time
            }, f'.tmp/cost_time.pt')

        elif backend == 'mindspore':
            from mindspore.ops import composite as C
            for _ in range(repeat_times):
                # forward
                forward_start_time = time.time()
                def flag_func():
                    pass
                out = F.linear(input_, weight, bias)
                print(out)
                forward_cost_time += (time.time() - forward_start_time)

                # backward
                backward_start_time = time.time()
                out.backward(torch.ones_like(out))
                backward_cost_time += (time.time() - backward_start_time)

            print(f'shape {shape} single forward cost time {forward_cost_time * 1000 / repeat_times} ms, '
                  f'single backward cost time {backward_cost_time * 1000 / repeat_times} ms')
            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.7:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.7), but got '
                    f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.7:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.7), but got '
                    f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")

    # set seed
    seed = 1921
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance'], help='test mode')
    parser.add_argument('--backend', type=str, choices=['torch', 'mindspore'], help='backend')
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestLinearFunctions().test_api_completeness()
    elif args.test_mode == 'performance':
        TestLinearFunctions().test_performance(args.backend, 100, (10, 10))
        TestLinearFunctions().test_performance(args.backend, 50, (100, 100))
        TestLinearFunctions().test_performance(args.backend, 50, (1000, 100))
        TestLinearFunctions().test_performance(args.backend, 50, (1000, 1000))