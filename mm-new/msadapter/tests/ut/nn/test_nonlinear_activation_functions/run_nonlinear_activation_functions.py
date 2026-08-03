import os
import time
import random
import argparse

import torch
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase


import numpy as np
class TestSigmoid(TestCase):

    def test_api_completeness(self):
        input_ = torch.randn(2)
        F.sigmoid(input_)


class TestSilu(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()

    def test_api_completeness_1(self):
        input_ = torch.randn(2)
        F.silu(input_, inplace=False)

    def test_api_completeness_2(self):
        input_ = torch.randn(2)
        F.silu(input_, inplace=True)


class TestSoftmax(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()
        self.test_api_completeness_3()
        self.test_api_completeness_4()
        self.test_api_completeness_5()

    def test_api_completeness_1(self):
        input_ = torch.randn(2, 3)
        dim = None
        dtype = None
        F.softmax(input=input_, dim=dim, dtype=dtype)

    def test_api_completeness_2(self):
        input_ = torch.randn(2, 3)
        dim = 1
        dtype = torch.float32
        F.softmax(input=input_, dim=dim, dtype=dtype)

    def test_api_completeness_3(self):
        input_ = torch.randn(2, 3)
        dim = 0
        dtype = torch.float64
        F.softmax(input=input_, dim=dim, dtype=dtype)

    def test_api_completeness_4(self):
        input_ = torch.randn(2, 3)
        dim = None
        dtype = torch.float16
        F.softmax(input=input_, dim=dim, dtype=dtype)

    def test_api_completeness_5(self):
        input_ = torch.randn(2, 3)
        dim = None
        dtype = torch.bfloat16
        F.softmax(input=input_, dim=dim, dtype=dtype)


class TestLogSigmoid(TestCase):

    def test_api_completeness(self):
        input_ = torch.randn(2)
        F.logsigmoid(input_)


class TestGelu(TestCase):

    def test_api_completeness(self):
        input_ = torch.randn((128, 256))
        F.gelu(input_)

        F.gelu(input=input_, approximate='none')

        input_ = input_.to(torch.bfloat16)
        F.gelu(input_, approximate='tanh')

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):
        input_ = torch.randn(shape, requires_grad=True).npu()
        # input_.requires_grad_(True)

        out = F.gelu(input_)    # pre-build

        forward_cost_time = 0
        backward_cost_time = 0

        pt_forward_time_diff_shape = {
            (10, 10): 5.8416295,
            (100, 100): 5.110542,
            (1000, 100): 3.4682989,
            (1000, 1000): 3.498404
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.535922,
            (100, 100): 0.3268861,
            (1000, 100): 2.2421765,
            (1000, 1000): 2.93197
        }

        if backend == 'torch':
            for _ in range(repeat_times):
                forward_start_time = time.time()
                out = F.gelu(input_)
                print(out)
                forward_cost_time += (time.time() - forward_start_time)

                out = out.sum()
                backward_start_time = time.time()
                out.backward()
                backward_cost_time += (time.time() - backward_start_time)
            print(f'shape {shape} single forward cost time {forward_cost_time * 1000 / repeat_times} ms, total {forward_cost_time} s '
                  f'\n single backward cost time {backward_cost_time * 1000 / repeat_times} ms, total {backward_cost_time} s')

            if not os.path.exists('.tmp'):
                os.mkdir('.tmp')
            torch.save({
                'forward_cost_time': forward_cost_time,
                'backward_cost_time': backward_cost_time
            }, f'.tmp/cost_time.pt')

        elif backend == 'mindspore':
            for _ in range(repeat_times):
                # forward
                forward_start_time = time.time()
                out = F.gelu(input_)
                forward_cost_time += (time.time() - forward_start_time)
                out = out.sum()

                # backward
                backward_start_time = time.time()
                out.backward()
                backward_cost_time += (time.time() - backward_start_time)

            print(f'shape {shape} single forward cost time {forward_cost_time * 1000 / repeat_times} ms, total {forward_cost_time} s'
                  f'\n single backward cost time {backward_cost_time * 1000 / repeat_times} ms, total {backward_cost_time} s')
            # pt_cost_time = torch.load(f'.tmp/cost_time.pt', map_location='cpu')
            if forward_cost_time * 1000 / repeat_times > pt_forward_time_diff_shape[shape] / 0.7:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.7), but got '
                    f'ms cost time {forward_cost_time * 1000 / repeat_times}, '
                    f'and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time * 1000 / repeat_times > pt_backward_time_diff_shape[shape] / 0.1:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time /  0.1), but got '
                    f'ms cost time {forward_cost_time * 1000 / repeat_times}, '
                    f'and pt cost time {pt_backward_time_diff_shape[shape]}.')


    def test_performance_with_mint(self, backend, repeat_times=1000, shape=(100, 100)):
        input_ = torch.randn(shape, requires_grad=True)

        mint_forward_cost_time = 1e-13
        mint_backward_cost_time = 1e-13
        forward_cost_time = 1e-13
        backward_cost_time = 1e-13

        if backend != 'mindspore':
            print("must in mindspore backend when compare with mint")
            return
        from mindspore import mint

        # mint
        for _ in range(repeat_times):
            # forward
            mint_forward_start_time = time.time()
            def flag_func():
                pass
            out = mint.nn.functional.gelu(input_)
            print(out)
            mint_forward_cost_time += (time.time() - mint_forward_start_time)

            # backward
            mint_backward_start_time = time.time()
            out.backward(torch.ones_like(out))
            mint_backward_cost_time += (time.time() - mint_backward_start_time)

        print(f'mint: shape {shape} single forward cost time {mint_forward_cost_time * 1000 / repeat_times} ms, total {mint_forward_cost_time} s'
            f'\n single backward cost time {mint_backward_cost_time * 1000 / repeat_times} ms, total {mint_backward_cost_time} s')

        for _ in range(repeat_times):
            # forward
            forward_start_time = time.time()
            def flag_func():
                pass
            out = F.gelu(input_)
            print(out)
            forward_cost_time += (time.time() - forward_start_time)

            # backward
            backward_start_time = time.time()
            out.backward(torch.ones_like(out))
            backward_cost_time += (time.time() - backward_start_time)

        print(f'msadapter: shape {shape} single forward cost time {forward_cost_time * 1000 / repeat_times} ms, total {forward_cost_time} s'
            f'\n single backward cost time {backward_cost_time * 1000 / repeat_times} ms, total {backward_cost_time} s')
        print(f'msadpater_forward_time / mint_forward_time = {forward_cost_time} / {mint_forward_cost_time} = {forward_cost_time / mint_forward_cost_time} times')
        print(f'msadpater_backward_time / mint_backward_time = {backward_cost_time} / {mint_backward_cost_time} = {backward_cost_time / mint_backward_cost_time} times')

        if forward_cost_time > mint_forward_cost_time * 1.3:
            raise ValueError(f'Expect msadapter forward cost time <= (mint cost time * 1.3), but got '
                f'msadapter cost time {forward_cost_time}, and mint cost time {mint_forward_cost_time}.')
        if backward_cost_time > mint_backward_cost_time * 1.6:
            raise ValueError(f'Expect msadapter backward cost time <= (mint cost time * 1.6, but got '
                f'msadapter cost time {backward_cost_time}, and mint cost time {mint_backward_cost_time}.')


class TestLogSoftmax(TestCase):

    def test_api_completeness(self):
        input_ = torch.randn((128, 256))
        F.log_softmax(input_)

        F.log_softmax(input=input_, dim=-1, dtype=None)

        F.log_softmax(input=input_, dtype=torch.bfloat16)

        input_ = torch.randn((128, 256)).to(torch.bfloat16)
        F.log_softmax(input=input_, dim=0)


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
        TestSigmoid().test_api_completeness()
        TestSilu().test_api_completeness()
        TestSoftmax().test_api_completeness()
        TestLogSigmoid().test_api_completeness()
        TestGelu().test_api_completeness()
        TestLogSoftmax().test_api_completeness()
    elif args.test_mode == 'performance':
        TestGelu().test_performance(args.backend, 100, (10, 10))
        TestGelu().test_performance(args.backend, 100, (100, 100))
        TestGelu().test_performance(args.backend, 100, (1000, 100))
        TestGelu().test_performance(args.backend, 100, (1000, 1000))
        if args.backend == "mindspore":
            TestGelu().test_performance_with_mint(args.backend, 100, (10, 10))
            TestGelu().test_performance_with_mint(args.backend, 100, (100, 100))
            TestGelu().test_performance_with_mint(args.backend, 100, (1000, 100))
            TestGelu().test_performance_with_mint(args.backend, 100, (1000, 1000))
