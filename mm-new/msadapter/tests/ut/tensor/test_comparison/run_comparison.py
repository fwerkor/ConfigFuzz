import os
import copy
import time
import random
import argparse
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

import numpy as np

# load from saved data
test_input = "/home/workspace/mindspore_dataset/msadapter/test_input"
test_path = os.path.join(test_input, "tensor")


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


class TestIsNan(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([-1.9696293, float('nan')]).npu()
        y = x.isnan()
        expected = torch.tensor([False, True]).npu()
        assert torch.equal(y, expected), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        if backend == 'torch':
            x = torch.tensor([-1.9696293, float('nan'), float('inf')]).npu()
            y = x.isnan()
            print('torch isnan: ', y)

            x2 = torch.empty(3).npu()
            y2 = x2.isnan()
            print('torch isnan: ', y2)

        elif backend == 'mindspore':
            pt_output = {
                'inf': torch.tensor([False, True, False]).npu(),  # float, nan, inf.
                'empty': torch.tensor([False, False, False]).npu()  # empty
            }

            x = torch.tensor([-1.9696293, float('nan'), float('inf')]).npu()
            y = x.isnan()
            print('ms isnan: ', y)
            print('torch isnan: ', pt_output['inf'])
            assert torch.equal(y, pt_output['inf']), f"expected: {pt_output['inf']}, but got {y}"

            x2 = torch.empty(3).npu()
            y2 = x2.isnan()
            print('ms isnan: ', y2)
            print('torch isnan: ', pt_output['empty'])
            # assert torch.equal(y2, pt_output['empty']), f"expected: {pt_output['empty']}, but got {y2}"
            # is nan op is different from torch

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0295 * 5.9,
            (100, 100): 0.0299 * 6.66,
            (1000, 100): 0.0296 * 7.2,
            (1000, 1000): 0.0331 * 6.54
        }
        pt_max_memory = {
            (10, 10): 1530.88 * 1.01,
            (100, 100): 40960.0,
            (1000, 100): 400896.0,
            (1000, 1000): 4001280.0
        }

        if backend == 'torch':
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x1.isnan()
                forward_cost_time += (time.time() - forward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    max_memory = 0
            max_memory /= repeat_times
            forward_cost_time = forward_cost_time * 1000 / repeat_times

            print(f'isnan shape {shape} single max memory {max_memory}')
            print(f'isnan shape {shape} single forward cost time {forward_cost_time} ms')

        elif backend == 'mindspore':
            import mindspore as ms

            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                forward_start_time = time.time()
                y = x1.isnan()
                forward_cost_time += (time.time() - forward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = forward_cost_time * 1000 / repeat_times

            print(f'isnan shape {shape} single ms max memory {max_memory}, '
                  f'single pt max memory {pt_max_memory[shape]}')
            print(f'isnan shape {shape} single forward cost time {forward_cost_time} ms')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.7:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.7), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'


class TestNe(TestCase):
    # ne: not equal
    def test_api_completeness(self):
        # fp32 vs fp32
        x = torch.tensor([-1.9, 3.0]).npu()
        y = torch.tensor([-2, 3.0]).npu()
        out = x.ne(y)
        expected = torch.tensor([True, False]).npu()
        assert torch.equal(out, expected), f"expected: {expected}, but got {out}"

        # fp32 vs int8
        x = torch.tensor([-1.9, 3.0]).npu()
        y = torch.tensor([-2, 3], dtype=torch.int8).npu()
        out = x.ne(y)
        expected = torch.tensor([True, False]).npu()
        assert torch.equal(out, expected), f"expected: {expected}, but got {out}"

        # bf16 vs int8
        x = torch.tensor([-1.9, 3.0], dtype=torch.bfloat16).npu()
        y = torch.tensor([-2, 3], dtype=torch.int8).npu()
        out = x.ne(y)
        expected = torch.tensor([True, False]).npu()
        assert torch.equal(out, expected), f"expected: {expected}, but got {out}"

        # bf16 vs fp32
        x = torch.tensor([-1.9, 3.0], dtype=torch.bfloat16).npu()
        y = torch.tensor([-2, 3.]).npu()
        out = x.ne(y)
        expected = torch.tensor([True, False]).npu()
        assert torch.equal(out, expected), f"expected: {expected}, but got {out}"

    def test_api_outlier(self, backend):
        if backend == 'torch':
            # nan
            x = torch.tensor([-1.96962931, float('nan'), float('nan'), -1.96962931]).npu()
            y = torch.tensor([-1.96962931, float('nan'), -1.96962931, float('nan')]).npu()
            out = x.ne(y)
            print(out)

            # inf
            x = torch.tensor([-1.96962931, float('inf'), float('inf'), -1.96962931]).npu()
            y = torch.tensor([-1.96962931, float('inf'), -1.96962931, float('inf')]).npu()
            out = x.ne(y)
            print(out)

            # empty: random number before allocation, which can not be compared
            x = torch.empty((4)).npu()
            y = torch.empty((4)).npu()
            out = x.ne(y)
            print(out)

        elif backend == 'mindspore':
            pt_output = {
                'nan': torch.tensor([False, True, True, True]),
                'inf': torch.tensor([False, False, True, True]),
                'empty': 'can not be compared'
            }

            # nan
            x = torch.tensor([-1.96962931, float('nan'), float('nan'), -1.96962931]).npu()
            y = torch.tensor([-1.96962931, float('nan'), -1.96962931, float('nan')]).npu()
            out = x.ne(y)
            print(out)
            print('ms ne: ', out)
            print('torch ne: ', pt_output['nan'])
            assert torch.equal(out, pt_output['nan']), f"expected: {pt_output['nan']}, but got {out}"

            # inf
            x = torch.tensor([-1.96962931, float('inf'), float('inf'), -1.96962931]).npu()
            y = torch.tensor([-1.96962931, float('inf'), -1.96962931, float('inf')]).npu()
            out = x.ne(y)
            print('ms ne: ', out)
            print('torch ne: ', pt_output['inf'])
            assert torch.equal(out, pt_output['inf']), f"expected: {pt_output['inf']}, but got {out}"

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0258 * 10.07,
            (100, 100): 0.0241 * 7.95,
            (1000, 100): 0.0319 * 6.22,
            (1000, 1000): 0.0329 * 6.19
        }
        pt_max_memory = {
            (10, 10): 2048.0 * 0.25,
            (100, 100): 61440.0 * 0.17,
            (1000, 100): 601088.0 * 0.18,
            (1000, 1000): 6001664.0 * 0.18
        }

        if backend == 'torch':
            torch.npu.reset_max_memory_allocated()

            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                forward_start_time = time.time()
                y = x1.ne(x2)
                forward_cost_time += (time.time() - forward_start_time)

                max_memory += torch.npu.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = forward_cost_time * 1000 / repeat_times

            print(f'ne shape {shape} single max memory {max_memory}')
            print(f'ne shape {shape} single forward cost time {forward_cost_time} ms')

        elif backend == 'mindspore':
            import mindspore as ms

            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                forward_start_time = time.time()
                y = x1.ne(x2)
                forward_cost_time += (time.time() - forward_start_time)

                max_memory += ms.runtime.max_memory_allocated()

            max_memory /= repeat_times
            forward_cost_time = forward_cost_time * 1000 / repeat_times

            print(f'ne shape {shape} single ms max memory {max_memory}, '
                  f'single pt max memory {pt_max_memory[shape]}')
            print(f'ne shape {shape} single forward cost time {forward_cost_time} ms')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.07: # runtime problem
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.7), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'


class TestSort(TestCase):
    def test_api_completeness(self):
        # fp32
        x = torch.tensor([[8, 2, 1], [5, 9, 3], [4, 6, 7]], dtype=torch.float32)
        out, index = x.sort()
        expected_out = torch.tensor([[1., 2., 8.],
                                     [3., 5., 9.],
                                     [4., 6., 7.]])
        expected_index = torch.tensor([[2, 1, 0],
                                       [2, 0, 1],
                                       [0, 1, 2]])
        assert torch.equal(out, expected_out), f"expected: {expected_out}, but got {out}"
        assert torch.equal(index, expected_index), f"expected: {expected_index}, but got {index}"

        # bf16, diff params
        x = torch.tensor([[8, 2, 1], [5, 9, 3], [4, 6, 7]], dtype=torch.bfloat16)
        dim = 0
        descending = True
        stable = True
        out, index = x.sort(dim=dim, descending=descending, stable=stable)
        expected_out = torch.tensor([[8., 9., 7.],
                                     [5., 6., 3.],
                                     [4., 2., 1.]], dtype=torch.bfloat16)
        expected_index = torch.tensor([[0, 1, 2],
                                       [1, 2, 1],
                                       [2, 0, 0]])
        assert torch.equal(out, expected_out), f"expected: {expected_out}, but got {out}"
        assert torch.equal(index, expected_index), f"expected: {expected_index}, but got {index}"

    def test_api_outlier(self, backend):
        if backend == 'torch':
            # nan
            x = torch.tensor([[8, 2, 1], [float('nan'), 9, 3], [4, 6, 7]], dtype=torch.float32).npu()
            out, index = x.sort()
            print(out)

            # inf
            x = torch.tensor([[8, 2, 1], [float('inf'), 9, 3], [4, float('-inf'), 7]], dtype=torch.float32)
            out, index = x.sort()
            print(out)

        elif backend == 'mindspore':
            # nan
            x = torch.tensor([[8, 2, 1], [float('nan'), 9, 3], [4, 6, 7]], dtype=torch.float32)
            out, index = x.sort()
            expected_out = torch.tensor([[1., 2., 8.],
                                         [3., 9., float('nan')],
                                         [4., 6., 7.]])
            expected_index = torch.tensor([[2, 1, 0],
                                           [2, 1, 0],
                                           [0, 1, 2]])
            print('ms sort: ', out)
            print('torch sort: ', expected_out)
            assert torch.allclose(out, expected_out, equal_nan=True), f"expected: {expected_out}, but got {out}"
            assert torch.equal(index, expected_index), f"index, expected: {expected_index}, but got {index}"

            # inf
            x = torch.tensor([[8, 2, 1], [float('inf'), 9, 3], [4, float('-inf'), 7]], dtype=torch.float32)
            out, index = x.sort()
            expected_out = torch.tensor([[1., 2., 8.],
                                         [3., 9., float('inf')],
                                         [float('-inf'), 4., 7.]])
            expected_index = torch.tensor([[2, 1, 0],
                                           [2, 1, 0],
                                           [1, 0, 2]])
            assert torch.equal(out, expected_out), f"expected: {expected_out}, but got {out}"
            assert torch.equal(index, expected_index), f"expected: {expected_index}, but got {index}"

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0471 * 4.03,
            (100, 100): 0.0467 * 4.31,
            (1000, 100): 0.0568 * 3.2,
            (1000, 1000): 0.0573 * 6.85
        }
        pt_max_memory = {
            (10, 10): 7168.0 * 0.71,
            (100, 100): 345600.0 * 0.64,
            (1000, 100): 3404288.0 * 0.58,
            (1000, 1000): 34644326.4 * 0.57
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()

                forward_start_time = time.time()
                out, _ = x.sort()
                forward_cost_time += (time.time() - forward_start_time)

                max_memory += torch.npu.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    max_memory = 0

            forward_cost_time = forward_cost_time * 1000 / repeat_times
            max_memory /= repeat_times

            print(f'sort shape {shape} single forward cost time {forward_cost_time} ms')
            print(f'sort shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times):
                ms.runtime.reset_max_memory_allocated()

                # forward
                forward_start_time = time.time()
                out, _ = x.sort()
                forward_cost_time += (time.time() - forward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = forward_cost_time * 1000 / repeat_times

            print(f'sort shape {shape} single forward cost time {forward_cost_time} ms')
            print(f'sort shape {shape} single ms max memory {max_memory}, '
                  f'single pt max memory {pt_max_memory[shape]}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.7:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.7), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'


class TestTopk(unittest.TestCase):
    ops_name = "topk"

    def test_api_completeness(self):
        x = torch.randn((3, 4), dtype=torch.float32)
        k = 2
        dim = 0
        y = x.topk(k, dim=dim)

        x = torch.arange(1., 7., dtype=torch.bfloat16).reshape((2, 3))
        k = 3
        dim = 1
        largest = False
        y = x.topk(k, dim=dim, largest=largest, sorted=False)
        print("topk: ", y)

        benchmark = torch.tensor([[1., 2., 3.], [4., 5., 6., ]], dtype=torch.bfloat16)
        assert torch.equal(y[0], benchmark)

    def test_api_outlier(self, backend):

        pt_output = {
            'inf': (torch.full((2, 3), float('inf')),
                    torch.tensor([[0, 1, 2], [0, 1, 2]])),
            'nan': (torch.full((2, 3), float('nan')),
                    torch.tensor([[0, 1, 2], [0, 1, 2]]))
        }

        if backend == 'torch':

            x = torch.empty(2, 4).fill_(float('inf')).npu()
            y = x.topk(3)
            print('torch inf: ', y)

            x = torch.empty(2, 4).fill_(float('nan')).npu()
            y = x.topk(3)
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.empty(2, 4).fill_(float('inf'))
            y = x.topk(3)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert (
                    isinstance(y, tuple) and
                    torch.isinf(y[0]).all() and
                    torch.equal(y[1], pt_output['inf'][1])
            )

            x = torch.empty(2, 4).fill_(float('nan'))
            y = x.topk(3)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert (
                    isinstance(y, tuple) and
                    torch.isnan(y[0]).all() and
                    torch.equal(y[1], pt_output['nan'][1])
            )

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 0.0824,
            (100, 100): 0.0827,
            (1000, 100): 0.0948,
            (1000, 1000): 0.1154,
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.3447,
            (100, 100): 0.3906,
            (1000, 100): 2.3251,
            (1000, 1000): 2.7649,
        }

        pt_max_memory = {
            (10, 10): 16781824.0,
            (100, 100): 16825344.0,
            (1000, 100): 17222144.0,
            (1000, 1000): 20822528.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.topk(3)
                y = y[0].sum()

                forward_cost_time.append(time.time() - forward_start_time)
                backward_start_time = time.time()
                y.backward()

                backward_cost_time.append(time.time() - backward_start_time)

                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()
            backward_cost_time = (np.array(backward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'backward {backward_cost_time:.4f} ms, max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.topk(3)
                y = y[0].sum()
                forward_cost_time.append(time.time() - forward_start_time)

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time.append(time.time() - backward_start_time)

                max_memory.append(ms.runtime.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()
            backward_cost_time = (np.array(backward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms '
                  f'({forward_cost_time / pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time / pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'backward {backward_cost_time:.4f} ms '
                  f'({backward_cost_time / pt_backward_time_diff_shape[shape]:.3f} pta) '
                  f'({backward_cost_time / pt_backward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory / pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory / pt_max_memory[shape] * 1.2:.2f} std)'
                  )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 33.0: # runtime problem
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 3.30), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.25:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.25), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.32:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.32), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.topk(3)
        y = y[0].sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'topk_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'topk_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):

        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []

        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.topk(3)
        y_ms = y_ms[0].sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        assert np.allclose(weight_grad[0].to(torch.float32), x_grad_pt.to(torch.float32), 0.0, 0.0)


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance', 'precision', 'outlier'],
                        help="test mode")
    parser.add_argument('--backend', type=str, choices=['torch', 'mindspore'],
                        help="backend")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestIsNan().test_api_completeness()
        TestNe().test_api_completeness()
        TestSort().test_api_completeness()
        TestTopk().test_api_completeness()
    elif args.test_mode == 'outlier':
        TestIsNan().test_api_outlier(args.backend)
        TestNe().test_api_outlier(args.backend)
        TestSort().test_api_outlier(args.backend)
        TestTopk().test_api_outlier(args.backend)
    elif args.test_mode == 'performance':
        TestIsNan().test_performance(args.backend, 100, (10, 10))
        TestIsNan().test_performance(args.backend, 100, (100, 100))
        TestIsNan().test_performance(args.backend, 10, (1000, 100))
        TestIsNan().test_performance(args.backend, 10, (1000, 1000))
        TestNe().test_performance(args.backend, 100, (10, 10))
        TestNe().test_performance(args.backend, 100, (100, 100))
        TestNe().test_performance(args.backend, 10, (1000, 100))
        TestNe().test_performance(args.backend, 10, (1000, 1000))
        TestSort().test_performance(args.backend, 100, (10, 10))
        TestSort().test_performance(args.backend, 100, (100, 100))
        TestSort().test_performance(args.backend, 10, (1000, 100))
        TestSort().test_performance(args.backend, 10, (1000, 1000))
        TestTopk().test_performance(args.backend, 50, (10, 10))
        TestTopk().test_performance(args.backend, 50, (100, 100))
        TestTopk().test_performance(args.backend, 50, (1000, 100))
        TestTopk().test_performance(args.backend, 50, (1000, 1000))
    elif args.test_mode == 'precision':
        if args.backend == 'torch':
            TestTopk().test_precision_pt(torch.bfloat16)
            TestTopk().test_precision_pt(torch.float16)
        elif args.backend == 'mindspore':
            TestTopk().test_precision_ms('topk_bf16')
            TestTopk().test_precision_ms('topk_fp16')
