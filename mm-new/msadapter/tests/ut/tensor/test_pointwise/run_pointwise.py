import os
import time
import random
import argparse
import torch
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


class TestAbs(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([-1, -2, 3])
        y = x.abs()
        print("abs: ", y)
        expected = torch.tensor([1, 2, 3])
        assert torch.equal(y, expected), f"for TestAbs, expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.abs()
            print('torch abs empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.abs()
            print('torch abs inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.abs()
            print('torch abs nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.abs()
            print('ms abs empty: ', y, ' torch abs empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.abs()
            print('ms abs inf: ', y, ' torch abs inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.abs()
            print('ms abs nan: ', y, ' torch abs nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0425 * 3.72,
            (100, 100): 0.0413 * 3.73,
            (1000, 100): 0.0561 * 2.59,
            (1000, 1000): 0.0626 * 2.3
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2245 * 1.14,
            (100, 100): 0.2772 * 0.9,
            (1000, 100): 2.1884 * 0.11,
            (1000, 1000): 3.4245 * 0.08
        }

        pt_max_memory = {
            (10, 10): 3584.0 * 0.57,
            (100, 100): 83456.0 * 0.74,
            (1000, 100): 802304.0 * 0.75,
            (1000, 1000): 8003072.0 * 0.75
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.abs()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'abs shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'abs shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                forward_start_time = time.time()
                y = x.abs()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'abs shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'abs shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.abs()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'abs_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'abs_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.abs()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestAbsolute(TestCase):
    def test_api_completeness(self):
        x = torch.tensor([-1, -2, 3])
        y = x.absolute()
        print("absolute: ", y)
        expected = torch.tensor([1, 2, 3])
        assert torch.equal(y, expected), f"for TestAbs, expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.absolute()
            print('torch absolute empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.absolute()
            print('torch absolute inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.absolute()
            print('torch absolute nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.absolute()
            print('ms absolute empty: ', y, ' torch absolute empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.absolute()
            print('ms absolute inf: ', y, ' torch absolute inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.absolute()
            print('ms absolute nan: ', y, ' torch absolute nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0505 * 3.34,
            (100, 100): 0.0518 * 3.2,
            (1000, 100): 0.0571 * 2.68,
            (1000, 1000): 0.063 * 2.43
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2755 * 1.03,
            (100, 100): 0.3563 * 0.78,
            (1000, 100): 2.5201 * 0.11,
            (1000, 1000): 2.9836 * 0.09
        }

        pt_max_memory = {
            (10, 10): 3584.0 * 0.57,
            (100, 100): 83456.0 * 0.74,
            (1000, 100): 802304.0 * 0.75,
            (1000, 1000): 8003072.0 * 0.75
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.absolute()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'absolute shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'absolute shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                forward_start_time = time.time()
                y = x.absolute()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'absolute shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'absolute shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.absolute()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")
        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'absolute_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'absolute_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.absolute()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestAdd(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([16.801167, 18.46366, 19.39963, 19.634846])
        y = x.add(20)
        print(y)
        expected = torch.tensor([36.801167, 38.46366, 39.39963, 39.634846])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for add, expected: {expected}, but got {y}"

        x1 = torch.tensor([0.7047168016, 0.6540585160, -0.3598807752, 0.0146041336])
        x2 = torch.tensor([[0.6164773703],
                           [0.1437176913],
                           [-0.1017297357],
                           [-0.5562295318]])
        y = x1.add(x2, alpha=10)  # auto broadcasting: a + b * alpha
        print(y)
        expected = torch.tensor([[6.8694906235, 6.8188323975, 5.8048930168, 6.1793780327],
                                 [2.1418936253, 2.0912353992, 1.0772961378, 1.4517810345],
                                 [-0.3125805557, -0.3632388413, -1.3771781921, -1.0026931763],
                                 [-4.8575782776, -4.9082369804, -5.9221758842, -5.5476913452]])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for TestAdd, expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.add(20)
            print('torch add empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.add(20)
            print('torch add inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.add(20)
            print('torch add nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.add(20)
            print('ms add empty: ', y, ' torch add empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.add(20)
            print('ms add inf: ', y, ' torch add inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.add(20)
            print('ms add nan: ', y, ' torch add nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1001, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0555 * 2.98,
            (100, 100): 0.0564 * 2.87,
            (1000, 100): 0.0636 * 2.37,
            (1000, 1000): 0.069 * 2.16
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2057 * 1.04,
            (100, 100): 0.2611 * 0.82,
            (1000, 100): 2.4216 * 0.09,
            (1000, 1000): 3.1956 * 0.07
        }

        pt_max_memory = {
            (10, 10): 3072.0 * 0.67,
            (100, 100): 43008.0 * 0.51,
            (1000, 100): 413696.0 * 0.51,
            (1000, 1000): 4014080.0 * 0.5
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.add(20)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'add shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'add shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward

                forward_start_time = time.time()
                y = x.add(20)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'add shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'add shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.add(20)
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0].contiguous(),
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")
        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'add_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'add_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.add(20)
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestCeil(TestCase):
    def test_api_completeness(self):
        x = torch.tensor([16.801167, 18.46366, 19.39963, 19.634846])
        y = x.ceil()
        print('ceil:', y)
        expected = torch.tensor([17., 19., 20., 20.])
        torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for ceil, expected: {expected}, but got {y}"

        x = torch.tensor([[-0.6341, -1.4208, -1.0900, 0.5826],
                          [-0.6341, -1.4208, -1.0900, 0.5826]])
        y = x.ceil()
        print('ceil:', y)
        expected = torch.tensor([[0., -1., -1., 1.],
                                 [0., -1., -1., 1.]])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for ceil, expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.ceil()
            print('torch ceil empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.ceil()
            print('torch ceil inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.ceil()
            print('torch ceil nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.ceil()
            print('ms ceil empty: ', y, ' torch ceil empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.ceil()
            print('ms ceil inf: ', y, ' torch ceil inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.ceil()
            print('ms ceil nan: ', y, ' torch ceil nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0409 * 3.91,
            (100, 100): 0.0402 * 3.91,
            (1000, 100): 0.0498 * 2.89,
            (1000, 1000): 0.057 * 2.51
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2037 * 1.22,
            (100, 100): 0.2585 * 0.95,
            (1000, 100): 2.4919 * 0.1,
            (1000, 1000): 3.2735 * 0.08
        }

        pt_max_memory = {
            (10, 10): 3072.0 * 0.67,
            (100, 100): 43008.0 * 0.96,
            (1000, 100): 413696.0 * 0.97,
            (1000, 1000): 4014080.0 * 1.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.ceil()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'ceil shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'ceil shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.ceil()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'ceil shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'ceil shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.ceil()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'ceil_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'ceil_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.ceil()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestClamp(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([-2.1907523, 0.1581391, 0.6457138, 0.3553900])
        y = x.clamp(min=-0.5, max=0.5)
        print(y)
        expected = torch.tensor([-0.5000000000, 0.1581391394, 0.5000000000, 0.3553900421])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for TestClamp, expected: {expected}, but got {y}"

        x = torch.tensor([2.3016286, 1.2361908, -0.1439718, 0.0350262])
        min_ = torch.linspace(-1, 1, steps=4)
        y = x.clamp(min=min_)
        print(y)
        expected = torch.tensor([2.3016286, 1.2361908, 0.3333333, 1.0000000])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for TestClamp, expected: {expected}, but got {y}"

        x = torch.tensor([-1.2858136, 0.4902603, 0.2593709, 0.7027668])
        max_ = torch.linspace(-1, 1, steps=4)
        y = x.clamp(max=max_)
        print(y)
        expected = torch.tensor([-1.2858136, -0.3333333, 0.2593709, 0.7027668])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for TestClamp, expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.tensor([[0.5000, 0.5000, 0.5000],
                                 [0.5000, 0.5000, 0.5000]]),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.clamp(min=-0.5, max=0.5)
            print('torch clamp empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.clamp(min=-0.5, max=0.5)
            print('torch clamp inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.clamp(min=-0.5, max=0.5)
            print('torch clamp nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.clamp(min=-0.5, max=0.5)
            print('ms clamp empty: ', y, ' torch clamp empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.clamp(min=-0.5, max=0.5)
            print('ms clamp inf: ', y, ' torch clamp inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.clamp(min=-0.5, max=0.5)
            print('ms clamp nan: ', y, ' torch clamp nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.057 * 3.75,
            (100, 100): 0.0589 * 3.63,
            (1000, 100): 0.0664 * 2.99,
            (1000, 1000): 0.0674 * 2.95
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.3309 * 1.11,
            (100, 100): 0.4364 * 0.84,
            (1000, 100): 3.0183 * 0.12,
            (1000, 1000): 3.1355 * 0.15
        }

        pt_max_memory = {
            (10, 10): 4608.0 * 0.76,
            (100, 100): 83968.0 * 0.63,
            (1000, 100): 803328.0 * 0.63,
            (1000, 1000): 8004096.0 * 0.63
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.clamp(min=-0.5, max=0.5)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'clamp shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'clamp shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward

                forward_start_time = time.time()
                y = x.clamp(min=-0.5, max=0.5)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()
                # backward

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'clamp shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'clamp shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.clamp(min=-0.5, max=0.5)
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'clamp_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'clamp_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.clamp(min=-0.5, max=0.5)
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestClip(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([-2.1907523, 0.1581391, 0.6457138, 0.3553900])
        y = x.clip(min=-0.5, max=0.5)
        print(y)
        expected = torch.tensor([-0.5000000000, 0.1581391394, 0.5000000000, 0.3553900421])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for TestClip, expected: {expected}, but got {y}"

        x = torch.tensor([2.3016286, 1.2361908, -0.1439718, 0.0350262])
        min_ = torch.linspace(-1, 1, steps=4)
        y = x.clip(min=min_)
        print(y)
        expected = torch.tensor([2.3016286, 1.2361908, 0.3333333, 1.0000000])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for TestClip, expected: {expected}, but got {y}"

        x = torch.tensor([-1.2858136, 0.4902603, 0.2593709, 0.7027668])
        max_ = torch.linspace(-1, 1, steps=4)
        y = x.clip(max=max_)
        print(y)
        expected = torch.tensor([-1.2858136, -0.3333333, 0.2593709, 0.7027668])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for TestClip, expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.tensor([[0.5000, 0.5000, 0.5000],
                                 [0.5000, 0.5000, 0.5000]]),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.clip(min=-0.5, max=0.5)
            print('torch clip empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.clip(min=-0.5, max=0.5)
            print('torch clip inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.clip(min=-0.5, max=0.5)
            print('torch clip nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.clip(min=-0.5, max=0.5)
            print('ms clip empty: ', y, ' torch clip empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.clip(min=-0.5, max=0.5)
            print('ms clip inf: ', y, ' torch clip inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.clip(min=-0.5, max=0.5)
            print('ms clip nan: ', y, ' torch clip nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0543 * 3.24,
            (100, 100): 0.055 * 3.15,
            (1000, 100): 0.0785 * 2.06,
            (1000, 1000): 0.0686 * 2.35
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.3104 * 1.11,
            (100, 100): 0.3651 * 0.94,
            (1000, 100): 2.1736 * 0.16,
            (1000, 1000): 3.4026 * 0.14
        }

        pt_max_memory = {
            (10, 10): 4608.0 * 0.78,
            (100, 100): 83968.0 * 0.63,
            (1000, 100): 803328.0 * 0.63,
            (1000, 1000): 8004096.0 * 0.63
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.clip(min=-0.5, max=0.5)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'clip shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'clip shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.clip(min=-0.5, max=0.5)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()
                # backward

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'clip shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'clip shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.clip(min=-0.5, max=0.5)
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'clip_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'clip_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.clip(min=-0.5, max=0.5)
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestCos(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([-0.2919261, 0.6896933, -0.8552471, 0.6651444])
        y = x.cos()
        expected = torch.tensor([0.9576913, 0.7714412, 0.6560320, 0.7868276])
        print(y)
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('nan')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.cos()
            print('torch cos empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.cos()
            print('torch cos inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.cos()
            print('torch cos nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.cos()
            print('ms cos empty: ', y, ' torch cos empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.cos()
            print('ms cos inf: ', y, ' torch cos inf: ', pt_output['inf'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['inf']), 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.cos()
            print('ms cos nan: ', y, ' torch cos nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0481 * 3.35,
            (100, 100): 0.0483 * 3.32,
            (1000, 100): 0.0555 * 2.71,
            (1000, 1000): 0.0592 * 2.58
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2591 * 1.06,
            (100, 100): 0.3729 * 0.8,
            (1000, 100): 2.5036 * 0.12,
            (1000, 1000): 3.1675 * 0.09
        }

        pt_max_memory = {
            (10, 10): 4096.0 * 0.62,
            (100, 100): 103936.0 * 0.79,
            (1000, 100): 1002496.0 * 0.8,
            (1000, 1000): 10003456.0 * 0.78
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.cos()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'cos shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'cos shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.cos()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'cos shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'cos shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.cos()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'cos_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'cos_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.cos()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestDiv(TestCase):

    def test_api_completeness(self):
        x1 = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
                           [0.1815, -1.0111, 0.9805, -1.5923],
                           [0.1062, 1.4581, 0.7759, -1.2344],
                           [-0.1830, -0.0313, 1.1908, -1.4757]])
        x2 = torch.tensor([0.8032, 0.2930, -0.8113, -0.2308])
        y = x1.div(x2)
        print(y)
        expected = torch.tensor([[-0.4620269, -6.6051192, 0.5676076, 1.2638649],
                                 [0.2259711, -3.4508533, -1.2085541, 6.8990469],
                                 [0.1322211, 4.9764500, -0.9563664, 5.3483539],
                                 [-0.2278386, -0.1068259, -1.4677677, 6.3938475]])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

        x1 = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
                           [0.1815, -1.0111, 0.9805, -1.5923],
                           [0.1062, 1.4581, 0.7759, -1.2344],
                           [-0.1830, -0.0313, 1.1908, -1.4757]])
        x2 = torch.tensor([0.8032, 0.2930, -0.8113, -0.2308])
        y = x1.div(x2, rounding_mode='floor')
        print(y)
        expected = torch.tensor([[-1., -7., 0., 1.],
                                 [0., -4., -2., 6.],
                                 [0., 4., -1., 5.],
                                 [-1., -1., -2., 6.]])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

        x1 = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
                           [0.1815, -1.0111, 0.9805, -1.5923],
                           [0.1062, 1.4581, 0.7759, -1.2344],
                           [-0.1830, -0.0313, 1.1908, -1.4757]])
        x2 = torch.tensor([0.8032, 0.2930, -0.8113, -0.2308])
        y = x1.div(x2, rounding_mode='trunc')
        print(y)
        expected = torch.tensor([[-0., -6., 0., 1.],
                                 [0., -3., -1., 6.],
                                 [0., 4., -0., 5.],
                                 [-0., -0., -1., 6.]])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x1 = torch.tensor([]).npu()
            x2 = 10
            y = x1.div(x2)
            print('torch div empty: ', y)

            x1 = torch.empty(2, 3).fill_(float('inf')).npu()
            x2 = 10
            y = x1.div(x2)
            print('torch div inf: ', y)

            x1 = torch.empty(2, 3).fill_(float('nan')).npu()
            x2 = 10
            y = x1.div(x2)
            print('torch div nan: ', y)

        elif backend == 'mindspore':
            x1 = torch.tensor([])
            x2 = 10
            y = x1.div(x2)
            print('ms div empty: ', y, ' torch div empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x1 = torch.empty(2, 3).fill_(float('inf'))
            x2 = 10
            y = x1.div(x2)
            print('ms div inf: ', y, ' torch div nan: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x1 = torch.empty(2, 3).fill_(float('nan'))
            x2 = 10
            y = x1.div(x2)
            print('ms div nan: ', y, ' torch div nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0588 * 3.36,
            (100, 100): 0.0593 * 3.3,
            (1000, 100): 0.0668 * 2.71,
            (1000, 1000): 0.0696 * 2.6
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.3663 * 0.94,
            (100, 100): 0.5137 * 0.66,
            (1000, 100): 3.957 * 0.09,
            (1000, 1000): 5.0489 * 0.07
        }

        pt_max_memory = {
            (10, 10): 4096.0 * 0.75,
            (100, 100): 123904.0 * 0.67,
            (1000, 100): 1202176.0 * 0.67,
            (1000, 1000): 12003328.0 * 0.67
        }

        if backend == 'torch':
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x1.div(x2)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()

                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'div shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'div shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x1.div(x2)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'div shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'div shape {shape} single ms max memory {max_memory}, '
                  f'single pt max memory {pt_max_memory[shape]}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x1 = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x2 = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x1.register_hook(x_hook)

        # forward and backward
        y = x1.div(x2)
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x1': x1,
            'x2': x2,
            'x1_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'div_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'div_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x1_pt, x2_pt, x1_grad_pt, y_pt = pt_data['x1'], pt_data['x2'], pt_data['x1_grad'], pt_data['y']

        print(f"pt grad is:\n{x1_grad_pt}")

        x1_pt.register_hook(weight_hook)

        # forward
        y_ms = x1_pt.div(x2_pt)
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x1_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x1_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestDivide(TestCase):

    def test_api_completeness(self):
        x1 = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
                           [0.1815, -1.0111, 0.9805, -1.5923],
                           [0.1062, 1.4581, 0.7759, -1.2344],
                           [-0.1830, -0.0313, 1.1908, -1.4757]])
        x2 = torch.tensor([0.8032, 0.2930, -0.8113, -0.2308])
        y = x1.divide(x2)
        print(y)
        expected = torch.tensor([[-0.4620269, -6.6051192, 0.5676076, 1.2638649],
                                 [0.2259711, -3.4508533, -1.2085541, 6.8990469],
                                 [0.1322211, 4.9764500, -0.9563664, 5.3483539],
                                 [-0.2278386, -0.1068259, -1.4677677, 6.3938475]])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x1 = torch.tensor([]).npu()
            x2 = 10
            y = x1.divide(x2)
            print('torch divide empty: ', y)

            x1 = torch.empty(2, 3).fill_(float('inf')).npu()
            x2 = 10
            y = x1.divide(x2)
            print('torch divide inf: ', y)

            x1 = torch.empty(2, 3).fill_(float('nan')).npu()
            x2 = 10
            y = x1.divide(x2)
            print('torch divide nan: ', y)

        elif backend == 'mindspore':
            x1 = torch.tensor([])
            x2 = 10
            y = x1.divide(x2)
            print('ms divide empty: ', y, ' torch divide empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x1 = torch.empty(2, 3).fill_(float('inf'))
            x2 = 10
            y = x1.divide(x2)
            print('ms divide inf: ', y, ' torch divide nan: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x1 = torch.empty(2, 3).fill_(float('nan'))
            x2 = 10
            y = x1.divide(x2)
            print('ms divide nan: ', y, ' torch divide nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0598 * 3.2,
            (100, 100): 0.0597 * 3.18,
            (1000, 100): 0.0743 * 2.38,
            (1000, 1000): 0.0688 * 2.55
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.3675 * 0.88,
            (100, 100): 0.5163 * 0.62,
            (1000, 100): 3.9985 * 0.08,
            (1000, 1000): 4.7376 * 0.07
        }

        pt_max_memory = {
            (10, 10): 4096.0 * 0.75,
            (100, 100): 123904.0 * 0.67,
            (1000, 100): 1202176.0 * 0.67,
            (1000, 1000): 12003328.0 * 0.67
        }

        if backend == 'torch':
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x1.divide(x2)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'divide shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'divide shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x1.divide(x2)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'divide shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'divide shape {shape} single ms max memory {max_memory}, '
                  f'single pt max memory {pt_max_memory[shape]}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x1 = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x2 = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x1.register_hook(x_hook)

        # forward and backward
        y = x1.divide(x2)
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x1': x1,
            'x2': x2,
            'x1_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'divide_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'divide_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x1_pt, x2_pt, x1_grad_pt, y_pt = pt_data['x1'], pt_data['x2'], pt_data['x1_grad'], pt_data['y']

        print(f"pt grad is:\n{x1_grad_pt}")

        x1_pt.register_hook(weight_hook)

        # forward
        y_ms = x1_pt.divide(x2_pt)
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x1_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x1_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestExp(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([0, 1])
        y = x.exp()
        print(y)
        expected = torch.tensor([1.0000000, 2.7182817])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.exp()
            print('torch exp empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.exp()
            print('torch exp inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.exp()
            print('torch exp nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.exp()
            print('ms exp empty: ', y, ' torch exp empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.exp()
            print('ms exp inf: ', y, ' torch exp inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.exp()
            print('ms exp nan: ', y, ' torch exp nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0392 * 4.32,
            (100, 100): 0.0369 * 4.51,
            (1000, 100): 0.0451 * 3.35,
            (1000, 1000): 0.0484 * 3.14
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2061 * 1.24,
            (100, 100): 0.2495 * 1.03,
            (1000, 100): 2.3236 * 0.11,
            (1000, 1000): 3.1326 * 0.08
        }

        pt_max_memory = {
            (10, 10): 3584.0 * 0.57,
            (100, 100): 83456.0 * 0.5,
            (1000, 100): 802304.0 * 0.5,
            (1000, 1000): 8003072.0 * 0.5
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.exp()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'exp shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'exp shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.exp()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'exp shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'exp shape {shape} single max memory {max_memory}')
            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.exp()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'exp_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'exp_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.exp()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestLog(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([3.6958766, 3.9205103, 0.8964190, 1.5646839, 2.9774647])
        y = x.log()
        print(y)
        expected = torch.tensor([1.3072177, 1.3662218, -0.1093474, 0.4476838, 1.0910722])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.log()
            print('torch log empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.log()
            print('torch log inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.log()
            print('torch log nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.log()
            print('ms log empty: ', y, ' torch log empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.log()
            print('ms log inf: ', y, ' torch log inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.log()
            print('ms log nan: ', y, ' torch log nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0364 * 4.46,
            (100, 100): 0.0376 * 4.24,
            (1000, 100): 0.0443 * 3.34,
            (1000, 1000): 0.0554 * 2.59
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.1914 * 1.36,
            (100, 100): 0.2404 * 1.08,
            (1000, 100): 2.2209 * 0.12,
            (1000, 1000): 4.2377 * 0.06
        }

        pt_max_memory = {
            (10, 10): 3072.0 * 0.67,
            (100, 100): 62976.0 * 0.66,
            (1000, 100): 602112.0 * 0.67,
            (1000, 1000): 6002688.0 * 0.67
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.log()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()

                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'log shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'log shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.log()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'log shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'log shape {shape} single max memory {max_memory}')
            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.tensor([1.0, 2.0, 4.0], dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.log()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'log_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'log_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.log()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestSin(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([0.7348099, 1.6324461, -1.5990942, -0.0387336])
        y = x.sin()
        print(y)
        expected = torch.tensor([0.6704461, 0.9981003, -0.9995996, -0.0387239])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('nan')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.sin()
            print('torch sin empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.sin()
            print('torch sin inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.sin()
            print('torch sin nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.sin()
            print('ms sin empty: ', y, ' torch sin empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.sin()
            print('ms sin inf: ', y, ' torch sin inf: ', pt_output['inf'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['inf']), 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.sin()
            print('ms sin nan: ', y, ' torch sin nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0424 * 3.78,
            (100, 100): 0.0413 * 3.83,
            (1000, 100): 0.0551 * 2.65,
            (1000, 1000): 0.0659 * 2.21
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2291 * 1.16,
            (100, 100): 0.2916 * 0.9,
            (1000, 100): 1.8826 * 0.14,
            (1000, 1000): 3.099 * 0.09
        }

        pt_max_memory = {
            (10, 10): 3584.0 * 0.57,
            (100, 100): 83456.0 * 0.74,
            (1000, 100): 802304.0 * 0.75,
            (1000, 1000): 8003072.0 * 0.75
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.sin()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'sin shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'sin shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.sin()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'sin shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'sin shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.sin()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'sin_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'sin_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.sin()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestSquare(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([-1.1704886, -0.5830599, 0.7463780, 0.3248016])
        y = x.square()
        print(y)
        expected = torch.tensor([1.3700435, 0.3399588, 0.5570801, 0.1054961])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.square()
            print('torch square empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.square()
            print('torch square inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.square()
            print('torch square nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.square()
            print('ms square empty: ', y, ' torch square empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.square()
            print('ms square inf: ', y, ' torch square inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.square()
            print('ms square nan: ', y, ' torch square nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.044 * 3.91,
            (100, 100): 0.0426 * 3.96,
            (1000, 100): 0.0629 * 2.45,
            (1000, 1000): 0.073 * 2.18
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2219 * 1.28,
            (100, 100): 0.2738 * 1.02,
            (1000, 100): 2.057 * 0.14,
            (1000, 1000): 3.4816 * 0.08
        }

        pt_max_memory = {
            (10, 10): 4096.0 * 0.62,
            (100, 100): 103936.0 * 0.6,
            (1000, 100): 1002496.0 * 0.6,
            (1000, 1000): 10003456.0 * 0.6
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.square()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'square shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'square shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.square()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'square shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'square shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.square()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'square_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'square_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.square()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestMul(TestCase):

    def test_api_completeness(self):

        x1 = torch.tensor([-1.9696293, 0.6183859, 0.8636860])
        x2 = 10
        y = x1.mul(x2)

        x1 = torch.tensor([[0.2036932],
                           [-1.4942740],
                           [1.1691655],
                           [0.4778146]])

        x2 = torch.tensor([[-0.0818304, 0.7099793, 0.1022753, 1.2349162]])
        y = x1.mul(x2)
        print('mul: ', y)
        expected = torch.tensor([[-0.0166683, 0.1446179, 0.0208328, 0.2515440],
                                 [0.1222771, -1.0609037, -0.1528273, -1.8453032],
                                 [-0.0956733, 0.8300833, 0.1195768, 1.4438214],
                                 [-0.0390998, 0.3392385, 0.0488686, 0.5900609]])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x1 = torch.tensor([]).npu()
            x2 = 10
            y = x1.mul(x2)
            print('torch mul empty: ', y)

            x1 = torch.empty(2, 3).fill_(float('inf')).npu()
            x2 = 10
            y = x1.mul(x2)
            print('torch mul inf: ', y)

            x1 = torch.empty(2, 3).fill_(float('nan')).npu()
            x2 = 10
            y = x1.mul(x2)
            print('torch mul nan: ', y)

        elif backend == 'mindspore':
            x1 = torch.tensor([])
            x2 = 10
            y = x1.mul(x2)
            print('ms mul empty: ', y, ' torch mul empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x1 = torch.empty(2, 3).fill_(float('inf'))
            x2 = 10
            y = x1.mul(x2)
            print('ms mul inf: ', y, ' torch mul nan: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x1 = torch.empty(2, 3).fill_(float('nan'))
            x2 = 10
            y = x1.mul(x2)
            print('ms mul nan: ', y, ' torch mul nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0437 * 4.27,
            (100, 100): 0.0444 * 4.19,
            (1000, 100): 0.0716 * 2.39,
            (1000, 1000): 0.0656 * 2.63
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2383 * 1.26,
            (100, 100): 0.3165 * 0.96,
            (1000, 100): 4.2418 * 0.07,
            (1000, 1000): 5.508 * 0.05
        }

        pt_max_memory = {
            (10, 10): 4096.0 * 0.5,
            (100, 100): 103936.0 * 0.6,
            (1000, 100): 1002496.0 * 0.6,
            (1000, 1000): 10003456.0 * 0.6
        }

        if backend == 'torch':
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x1.mul(x2)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'mul shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'mul shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward

                forward_start_time = time.time()
                y = x1.mul(x2)
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'mul shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'mul shape {shape} single ms max memory {max_memory}, '
                  f'single pt max memory {pt_max_memory[shape]}')
            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x1 = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x2 = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x1.register_hook(x_hook)

        # forward and backward
        y = x1.mul(x2)
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x1': x1,
            'x2': x2,
            'x1_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'mul_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'mul_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x1_pt, x2_pt, x1_grad_pt, y_pt = pt_data['x1'], pt_data['x2'], pt_data['x1_grad'], pt_data['y']

        print(f"pt grad is:\n{x1_grad_pt}")

        x1_pt.register_hook(weight_hook)

        # forward
        y_ms = x1_pt.mul(x2_pt)
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x1_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x1_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestDim(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = x.dim()
        print(y)
        expected = 2
        assert y == expected, f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': 1,
            'inf': 2,
            'nan': 2
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.dim()
            print('torch dim empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.dim()
            print('torch dim inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.dim()
            print('torch dim nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.dim()
            print('ms dim empty: ', y, ' torch dim empty: ', pt_output['empty'])
            assert y == pt_output['empty']

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.dim()
            print('ms dim inf: ', y, ' torch dim inf: ', pt_output['inf'])
            assert y == pt_output['inf']

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.dim()
            print('ms dim nan: ', y, ' torch dim nan: ', pt_output['nan'])
            assert y == pt_output['nan']

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0007 * 162.71,
            (100, 100): 0.0007 * 170.43,
            (1000, 100): 0.0008 * 122.25,
            (1000, 1000): 0.0009 * 107.89
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                forward_start_time = time.time()
                y = x.dim()
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'dim shape {shape} single forward cost time {forward_cost_time} ms')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):

                # forward
                forward_start_time = time.time()
                y = x.dim()
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'dim shape {shape} single forward cost time {forward_cost_time} ms')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')


class TestFloor(TestCase):
    def test_api_completeness(self):
        x = torch.tensor([16.801167, 18.46366, 19.39963, 19.634846])
        y = x.floor()
        print('floor:', y)
        expected = torch.tensor([16., 18., 19., 19.])
        torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for floor, expected: {expected}, but got {y}"

        x = torch.tensor([[-0.6341, -1.4208, -1.0900, 0.5826],
                          [-0.6341, -1.4208, -1.0900, 0.5826]])
        y = x.floor()
        print('floor:', y)
        expected = torch.tensor([[-1., -2., -2., 0.],
                                 [-1., -2., -2., 0.]])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"for floor, expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.floor()
            print('torch floor empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.floor()
            print('torch floor inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.floor()
            print('torch floor nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.floor()
            print('ms floor empty: ', y, ' torch floor empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.floor()
            print('ms floor inf: ', y, ' torch floor inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.floor()
            print('ms floor nan: ', y, ' torch floor nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0418 * 3.68,
            (100, 100): 0.0414 * 3.63,
            (1000, 100): 0.0661 * 2.13,
            (1000, 1000): 0.0727 * 1.94
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2082 * 1.2,
            (100, 100): 0.2635 * 0.94,
            (1000, 100): 2.5752 * 0.09,
            (1000, 1000): 3.053 * 0.08
        }

        pt_max_memory = {
            (10, 10): 3072.0 * 0.67,
            (100, 100): 43008.0 * 0.51,
            (1000, 100): 413696.0 * 0.51,
            (1000, 1000): 4014080.0 * 0.5
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.floor()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'floor shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'floor shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.floor()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'floor shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'floor shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.floor()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'floor_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'floor_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.floor()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestItem(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([[1.2]])
        y = x.item()
        print(y)
        expected = 1.2
        assert round(y, 1) == expected, f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'inf': float('inf'),
            'nan': float('nan'),
        }

        if backend == 'torch':
            x = torch.tensor(float('inf')).npu()
            y = x.item()
            print('torch item inf: ', y)

            x = torch.tensor(float('nan')).npu()
            y = x.item()
            print('torch item nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor(float('inf')).npu()
            y = x.item()
            print('ms item inf: ', y, ' torch item inf: ', pt_output['inf'])
            assert y == pt_output['inf']

            x = torch.tensor(float('nan')).npu()
            y = x.item()
            print('ms item nan: ', y, ' torch item nan: ', pt_output['nan'])
            import math
            assert math.isnan(y)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0175 * 6.45,
            (100, 100): 0.0173 * 7.08,
            (1000, 100): 0.0172 * 7.59,
            (1000, 1000): 0.0175 * 7.67
        }

        if backend == 'torch':
            x = torch.tensor([[1.2]], dtype=torch.bfloat16).npu()

            for i in range(repeat_times + 1):
                forward_start_time = time.time()
                y = x.item()
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'item shape {shape} single forward cost time {forward_cost_time} ms')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.tensor([[1.2]], dtype=torch.bfloat16).npu()

            for i in range(repeat_times + 1):

                # forward
                forward_start_time = time.time()
                y = x.item()
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'item shape {shape} single forward cost time {forward_cost_time} ms')
            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')


class TestNelement(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([[1.2, 1.0, 0],
                          [-10.333, -2, 6]])
        y = x.nelement()
        print(y)
        expected = 6
        assert y == expected, f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': 0,
            'inf': 6,
            'nan': 6
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.nelement()
            print('torch nelement empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.nelement()
            print('torch nelement inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.nelement()
            print('torch nelement nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.nelement()
            print('ms nelement empty: ', y, ' torch nelement empty: ', pt_output['empty'])
            assert y == pt_output['empty']

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.nelement()
            print('ms nelement inf: ', y, ' torch nelement inf: ', pt_output['inf'])
            assert y == pt_output['inf']

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.nelement()
            print('ms nelement nan: ', y, ' torch nelement nan: ', pt_output['nan'])
            assert y == pt_output['nan']

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0003 * 620.67,
            (100, 100): 0.0004 * 493.75,
            (1000, 100): 0.0004 * 431.25,
            (1000, 1000): 0.0004 * 428.75
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                forward_start_time = time.time()
                y = x.nelement()
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'nelement shape {shape} single forward cost time {forward_cost_time} ms')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.nelement()
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'nelement shape {shape} single forward cost time {forward_cost_time} ms')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')


class TestNumel(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([[1.2, 1.0, 0],
                          [-10.333, -2, 6]])
        y = x.numel()
        print(y)
        expected = 6
        assert y == expected, f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': 0,
            'inf': 6,
            'nan': 6
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.numel()
            print('torch numel empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.numel()
            print('torch numel inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.numel()
            print('torch numel nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.numel()
            print('ms numel empty: ', y, ' torch numel empty: ', pt_output['empty'])
            assert y == pt_output['empty']

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.numel()
            print('ms numel inf: ', y, ' torch numel inf: ', pt_output['inf'])
            assert y == pt_output['inf']

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.numel()
            print('ms numel nan: ', y, ' torch numel nan: ', pt_output['nan'])
            assert y == pt_output['nan']

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0003 * 602.33,
            (100, 100): 0.0003 * 635.67,
            (1000, 100): 0.0004 * 417.0,
            (1000, 1000): 0.0004 * 412.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                forward_start_time = time.time()
                y = x.numel()
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'numel shape {shape} single forward cost time {forward_cost_time} ms')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.numel()
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'numel shape {shape} single forward cost time {forward_cost_time} ms')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')


class TestSize(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([[1.2, 1.0, 0],
                          [-10.333, -2, 6]])
        y = x.size()
        print(y)
        expected = torch.Size([2, 3])
        assert y == expected, f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.Size([0]),
            'inf': torch.Size([2, 3]),
            'nan': torch.Size([2, 3])
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.size()
            print('torch size empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.size()
            print('torch size inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.size()
            print('torch size nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.size()
            print('ms size empty: ', y, ' torch size empty: ', pt_output['empty'])
            assert y == pt_output['empty']

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.size()
            print('ms size inf: ', y, ' torch size inf: ', pt_output['inf'])
            assert y == pt_output['inf']

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.size()
            print('ms size nan: ', y, ' torch size nan: ', pt_output['nan'])
            assert y == pt_output['nan']

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0006 * 196.5,
            (100, 100): 0.0005 * 249.2,
            (1000, 100): 0.0006 * 172.67,
            (1000, 1000): 0.0007 * 151.43
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                forward_start_time = time.time()
                y = x.size()
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'size shape {shape} single forward cost time {forward_cost_time} ms')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.size()
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'size shape {shape} single forward cost time {forward_cost_time} ms')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')


class TestRound(TestCase):

    def test_api_completeness(self):
        decimals = 0
        x = torch.tensor([0.8, 1.5, 2.3, 2.5, -4.5])
        y = x.round(decimals=decimals)
        print(y)
        expected = torch.tensor([1., 2., 2., 2., -4.])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

        decimals = 1
        x = torch.tensor([0.81, 1.52, 2.35, 2.53, -4.57])
        y = x.round(decimals=decimals)
        print(y)
        expected = torch.tensor([0.8, 1.5, 2.4, 2.5, -4.6])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

        decimals = -1
        x = torch.tensor([10.8, 11.5, 21.3, 22.5, -41.5])
        y = x.round(decimals=decimals)
        print(y)
        expected = torch.tensor([10., 10., 20., 20., -40.])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.round()
            print('torch round empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.round()
            print('torch round inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.round()
            print('torch round nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.round()
            print('ms round empty: ', y, ' torch round empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.round()
            print('ms round inf: ', y, ' torch round inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.round()
            print('ms round nan: ', y, ' torch round nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0417 * 4.01,
            (100, 100): 0.0414 * 3.9,
            (1000, 100): 0.0595 * 2.48,
            (1000, 1000): 0.0618 * 2.38
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2022 * 1.23,
            (100, 100): 0.2517 * 0.98,
            (1000, 100): 2.6253 * 0.09,
            (1000, 1000): 3.2127 * 0.08
        }

        pt_max_memory = {
            (10, 10): 3072.0 * 0.67,
            (100, 100): 43008.0 * 0.96,
            (1000, 100): 413696.0 * 0.97,
            (1000, 1000): 4014080.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.round()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'round shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'round shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.round()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'round shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'round shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.round()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'round_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'round_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.round()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestSqrt(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([16, 0.16])
        y = x.sqrt()
        print(y)
        expected = torch.tensor([4., 0.4])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.sqrt()
            print('torch sqrt empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.sqrt()
            print('torch sqrt inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.sqrt()
            print('torch sqrt nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.sqrt()
            print('ms sqrt empty: ', y, ' torch sqrt empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.sqrt()
            print('ms sqrt inf: ', y, ' torch sqrt inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.sqrt()
            print('ms sqrt nan: ', y, ' torch sqrt nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0449 * 3.8,
            (100, 100): 0.0464 * 3.49,
            (1000, 100): 0.0541 * 2.74,
            (1000, 1000): 0.0664 * 2.24
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2422 * 1.26,
            (100, 100): 0.3917 * 0.79,
            (1000, 100): 2.3545 * 0.13,
            (1000, 1000): 3.7846 * 0.08
        }

        pt_max_memory = {
            (10, 10): 4096.0 * 0.5,
            (100, 100): 143872.0 * 0.29,
            (1000, 100): 1402880.0 * 0.29,
            (1000, 1000): 14003200.0 * 0.29
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.sqrt()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'sqrt shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'sqrt shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.sqrt()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'sqrt shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'sqrt shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.tensor([[16., 0.16],
                          [4., 2.]], dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.sqrt()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'sqrt_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'sqrt_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.sqrt()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        self.assertEqual(y_ms, y_pt), "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestSub(TestCase):

    def test_api_completeness(self):

        x1 = torch.tensor([4., 5., 6.])
        x2 = torch.tensor([4., 5., 6.])
        a = 0.5
        y = x1.sub(x2, alpha=a)

        x1 = torch.tensor([4., 5., 6.])
        x2 = torch.tensor([1., 2., 3.])
        a = 2
        y = x1.sub(x2, alpha=a)
        print('sub: ', y)

        expected = torch.tensor([2., 1., 0, ])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x1 = torch.tensor([]).npu()
            x2 = 10
            y = x1.sub(x2)
            print('torch sub empty: ', y)

            x1 = torch.empty(2, 3).fill_(float('inf')).npu()
            x2 = 10
            y = x1.sub(x2)
            print('torch sub inf: ', y)

            x1 = torch.empty(2, 3).fill_(float('nan')).npu()
            x2 = 10
            y = x1.sub(x2)
            print('torch sub nan: ', y)

        elif backend == 'mindspore':
            x1 = torch.tensor([])
            x2 = 10
            y = x1.sub(x2)
            print('ms sub empty: ', y, ' torch sub empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x1 = torch.empty(2, 3).fill_(float('inf'))
            x2 = 10
            y = x1.sub(x2)
            print('ms sub inf: ', y, ' torch sub nan: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x1 = torch.empty(2, 3).fill_(float('nan'))
            x2 = 10
            y = x1.sub(x2)
            print('ms sub nan: ', y, ' torch sub nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0535 * 3.63,
            (100, 100): 0.0539 * 3.79,
            (1000, 100): 0.0757 * 2.43,
            (1000, 1000): 0.065 * 3.23
        }

        pt_max_memory = {
            (10, 10): 3584.0 * 0.14,
            (100, 100): 83456.0 * 0.25,
            (1000, 100): 802304.0 * 0.25,
            (1000, 1000): 8003072.0 * 0.25
        }

        if backend == 'torch':
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x1.sub(x2)
                forward_cost_time += (time.time() - forward_start_time)

                max_memory += torch.npu.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)

            print(f'sub shape {shape} single forward cost time {forward_cost_time} ms')
            print(f'sub shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                forward_start_time = time.time()
                y = x1.sub(x2)
                forward_cost_time += (time.time() - forward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)

            print(f'sub shape {shape} single forward cost time {forward_cost_time} ms')
            print(f'sub shape {shape} single ms max memory {max_memory}, '
                  f'single pt max memory {pt_max_memory[shape]}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):

        x1 = torch.randn((100, 100), dtype=torch.bfloat16, requires_grad=True).npu()
        x2 = torch.randn((100, 100), dtype=torch.bfloat16, requires_grad=True).npu()

        # forward and backward
        y = x1.sub(x2)

        # save inputs, outputs and grad
        save_dict = {
            'x1': x1,
            'x2': x2,
            'y': y
        }

        print(f"save_dict: {save_dict}")

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'sub_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'sub_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x1_pt, x2_pt, y_pt = pt_data['x1'], pt_data['x2'], pt_data['y']

        # forward
        def flag_func():
            pass

        y_ms = x1_pt.sub(x2_pt)

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestTrunc(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([0.8, 1.5, 2.3, 2.5, -4.5])
        y = x.trunc()
        print(y)
        expected = torch.tensor([0., 1., 2., 2., -4.])
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.trunc()
            print('torch trunc empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.trunc()
            print('torch trunc inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.trunc()
            print('torch trunc nan: ', y)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.trunc()
            print('ms trunc empty: ', y, ' torch trunc empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.trunc()
            print('ms trunc inf: ', y, ' torch trunc inf: ', pt_output['inf'])
            assert np.allclose(y, pt_output['inf'], 0.0, 0.0)

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.trunc()
            print('ms trunc nan: ', y, ' torch trunc nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        backward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0441 * 3.61,
            (100, 100): 0.0438 * 3.58,
            (1000, 100): 0.0526 * 2.88,
            (1000, 1000): 0.0612 * 2.37
        }
        pt_backward_time_diff_shape = {
            (10, 10): 0.2244 * 1.13,
            (100, 100): 0.2843 * 0.89,
            (1000, 100): 1.9566 * 0.13,
            (1000, 1000): 3.6022 * 0.07
        }

        pt_max_memory = {
            (10, 10): 3072.0 * 0.67,
            (100, 100): 43008.0 * 0.51,
            (1000, 100): 413696.0 * 0.51,
            (1000, 1000): 4014080.0 * 0.5
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.trunc()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += torch.npu.max_memory_allocated()

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'trunc shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'trunc shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.trunc()
                forward_cost_time += (time.time() - forward_start_time)

                y = y.sum()

                # backward
                backward_start_time = time.time()
                y.backward()
                backward_cost_time += (time.time() - backward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    backward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            backward_cost_time = round(backward_cost_time * 1000 / repeat_times, 4)

            print(f'trunc shape {shape} single forward cost time {forward_cost_time} ms, '
                  f'single backward cost time {backward_cost_time} ms')

            print(f'trunc shape {shape} single max memory {max_memory}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            if backward_cost_time > pt_backward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms backward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {backward_cost_time}, and pt cost time {pt_backward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.trunc()
        y = y.sum()
        y.backward()

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'x_grad': x_grad[0],
            'y': y
        }

        print(f"save_dict: {save_dict}")
        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'trunc_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'trunc_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x_pt, x_grad_pt, y_pt = pt_data['x'], pt_data['x_grad'], pt_data['y']

        x_pt.register_hook(weight_hook)

        # forward
        y_ms = x_pt.trunc()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        if np.allclose(weight_grad[0].to(torch.float16), x_grad_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestType(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([0.8, 1.5, 2.3, 2.5, -4.5], dtype=torch.float16)
        y = x.type()
        print(y)

        x = torch.tensor([0.8, 1.5, 2.3, 2.5, -4.5], dtype=torch.float16)
        dtype = torch.bfloat16
        y = x.type(dtype=dtype)
        print(y.dtype)
        assert dtype == y.dtype, f"expected: {dtype}, but got {y.dtype}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.bfloat16,
            'inf': torch.bfloat16,
            'nan': torch.bfloat16
        }

        if backend == 'torch':
            x = torch.tensor([]).npu()
            y = x.type(dtype=torch.bfloat16)
            print('torch type empty: ', y.dtype)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.type(dtype=torch.bfloat16)
            print('torch type inf: ', y.dtype)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.type(dtype=torch.bfloat16)
            print('torch type nan: ', y.dtype)

        elif backend == 'mindspore':
            x = torch.tensor([])
            y = x.type(dtype=torch.bfloat16)
            print('ms type empty: ', y.dtype, ' torch type empty: ', pt_output['empty'])
            assert y.dtype == pt_output['empty']

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.type(dtype=torch.bfloat16)
            print('ms type inf: ', y.dtype, ' torch type inf: ', pt_output['inf'])
            assert y.dtype == pt_output['inf']

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.type(dtype=torch.bfloat16)
            print('ms type nan: ', y.dtype, ' torch type nan: ', pt_output['nan'])
            assert y.dtype == pt_output['nan']

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0238 * 7.05,
            (100, 100): 0.0231 * 7.71,
            (1000, 100): 0.0336 * 4.81,
            (1000, 1000): 0.0333 * 5.72
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):
                forward_start_time = time.time()
                y = x.type(dtype=torch.float16)
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'type shape {shape} single forward cost time {forward_cost_time} ms')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for i in range(repeat_times + 1):

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x.type(dtype=torch.float16)
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'type shape {shape} single forward cost time {forward_cost_time} ms')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')


class TestTypeas(TestCase):

    def test_api_completeness(self):
        x1 = torch.randn((2, 2), dtype=torch.float16)
        x2 = torch.randn((1, 2), dtype=torch.bfloat16)
        y = x1.type_as(x2)
        print(y.dtype)
        expected = torch.bfloat16
        assert expected == y.dtype, f"expected: {expected}, but got {y.dtype}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.bfloat16,
            'inf': torch.bfloat16,
            'nan': torch.bfloat16
        }

        if backend == 'torch':
            x2 = torch.randn((1, 2), dtype=torch.bfloat16)
            x1 = torch.tensor([]).npu()
            y = x1.type_as(x2)
            print('torch type_as empty: ', y.dtype)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x1.type_as(x2)
            print('torch type_as inf: ', y.dtype)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x1.type_as(x2)
            print('torch type_as nan: ', y.dtype)

        elif backend == 'mindspore':
            x2 = torch.randn((1, 2), dtype=torch.bfloat16)
            x1 = torch.tensor([])
            y = x1.type_as(x2)
            print('ms type_as empty: ', y, ' torch type_as empty: ', pt_output['empty'])
            assert y.dtype == pt_output['empty']

            x1 = torch.empty(2, 3).fill_(float('inf'))
            y = x1.type_as(x2)
            print('ms type_as inf: ', y, ' torch type_as inf: ', pt_output['inf'])
            assert y.dtype == pt_output['inf']

            x1 = torch.empty(2, 3).fill_(float('nan'))
            y = x1.type_as(x2)
            print('ms type_as nan: ', y, ' torch type_as nan: ', pt_output['nan'])
            assert y.dtype == pt_output['nan']

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0199 * 8.13,
            (100, 100): 0.0199 * 8.74,
            (1000, 100): 0.0268 * 5.89,
            (1000, 1000): 0.0328 * 5.54
        }

        if backend == 'torch':
            x1 = torch.randn(shape, dtype=torch.float16, requires_grad=True).npu()
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            for i in range(repeat_times + 1):
                forward_start_time = time.time()
                y = x1.type_as(x2)
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'type_as shape {shape} single forward cost time {forward_cost_time} ms')

        elif backend == 'mindspore':
            import mindspore as ms
            x1 = torch.randn(shape, dtype=torch.float16, requires_grad=True).npu()
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            for i in range(repeat_times + 1):

                # forward
                def flag_func():
                    pass

                forward_start_time = time.time()
                y = x1.type_as(x2)
                forward_cost_time += (time.time() - forward_start_time)

                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0

            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'type_as shape {shape} single forward cost time {forward_cost_time} ms')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')


class TestMatmul(TestCase):

    def test_api_completeness(self):

        x1 = torch.randn(3, dtype=torch.float16)
        x2 = torch.randn(3, dtype=torch.float16)
        y = x1.matmul(x2)

        x1 = torch.randn(10, 3, 4, dtype=torch.float16)
        x2 = torch.randn(4, dtype=torch.float16)
        y = x1.matmul(x2)

        x1 = torch.reshape(torch.arange(2 * 3 * 4, dtype=torch.float16), (2, 3, 4))
        x2 = torch.reshape(torch.arange(4 * 5, dtype=torch.float16), (4, 5))
        y = x1.matmul(x2)
        print('matmul: ', y)
        expected = torch.tensor([[[70., 76., 82., 88., 94.],
                                  [190., 212., 234., 256., 278.],
                                  [310., 348., 386., 424., 462.]],
                                 [[430., 484., 538., 592., 646.],
                                  [550., 620., 690., 760., 830.],
                                  [670., 756., 842., 928., 1014.]]], dtype=torch.float16)
        assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6), f"expected: {expected}, but got {y}"

    def test_api_outlier(self, backend):
        pt_output = {
            'empty': torch.tensor([0., 0., 0.]),
            'nan': torch.tensor([float('nan'), float('nan')])
        }

        if backend == 'torch':
            x1 = torch.tensor([[], [], []], dtype=torch.float16).npu()
            x2 = torch.tensor([], dtype=torch.float16).npu()
            y = x1.matmul(x2)
            print('torch matmul empty: ', y)

            x1 = torch.empty(2, 3).fill_(float('nan')).npu()
            x2 = torch.randn(3, dtype=torch.float16).npu()
            y = x1.matmul(x2)
            print('torch matmul nan: ', y)

        elif backend == 'mindspore':
            x1 = torch.tensor([[], [], []], dtype=torch.float16)
            x2 = torch.tensor([], dtype=torch.float16)
            y = x1.matmul(x2)
            print('ms matmul empty: ', y, ' torch matmul empty: ', pt_output['empty'])
            assert np.allclose(y, pt_output['empty'], 0.0, 0.0)

            x1 = torch.empty(2, 3, dtype=torch.float16).fill_(float('nan'))
            x2 = torch.randn(3, dtype=torch.float16)
            y = x1.matmul(x2)
            print('ms matmul nan: ', y, ' torch matmul nan: ', pt_output['nan'])
            assert np.allclose(torch.isnan(y), torch.isnan(pt_output['nan']), 0.0, 0.0)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = 0
        max_memory = 0

        pt_forward_time_diff_shape = {
            (10, 10): 0.0561 * 3.14,
            (100, 100): 0.0553 * 3.41,
            (1000, 100): 0.0824 * 2.87,
            (1000, 1000): 0.0788 * 2.64
        }

        pt_max_memory = {
            (10, 10): 4096.0 * 0.12,
            (100, 100): 103936.0 * 0.2,
            (1000, 100): 5061632.0 * 0.89,
            (1000, 1000): 10003456.0 * 0.2
        }

        if backend == 'torch':
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).t().npu()
            for i in range(repeat_times + 1):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x1.matmul(x2)
                forward_cost_time += (time.time() - forward_start_time)

                max_memory += torch.npu.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)
            print(f'matmul shape {shape} single forward cost time {forward_cost_time} ms')
            print(f'matmul shape {shape} single max memory {max_memory}')

        elif backend == 'mindspore':
            import mindspore as ms
            x1 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
            x2 = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).t()

            for i in range(repeat_times + 1):
                ms.runtime.reset_max_memory_allocated()

                # forward
                forward_start_time = time.time()
                y = x1.matmul(x2)
                forward_cost_time += (time.time() - forward_start_time)

                max_memory += ms.runtime.max_memory_allocated()
                # 跳过第一次
                if i == 0:
                    forward_cost_time = 0
                    max_memory = 0

            max_memory /= repeat_times
            forward_cost_time = round(forward_cost_time * 1000 / repeat_times, 4)

            print(f'matmul shape {shape} single forward cost time {forward_cost_time} ms')

            print(f'matmul shape {shape} single ms max memory {max_memory}, '
                  f'single pt max memory {pt_max_memory[shape]}')

            if forward_cost_time > pt_forward_time_diff_shape[shape] / 0.8:
                raise ValueError(f'Expect ms forward cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {forward_cost_time}, and pt cost time {pt_forward_time_diff_shape[shape]}.')
            assert max_memory < pt_max_memory[shape] / 0.8, f'out of memory threshold.'

    def test_precision_pt(self, dtype):

        x1 = torch.randn((10, 100), dtype=torch.bfloat16, requires_grad=True).npu()
        x2 = torch.randn((100, 10), dtype=torch.bfloat16, requires_grad=True).npu()

        # forward and backward
        y = x1.matmul(x2)

        # save inputs, outputs and grad
        save_dict = {
            'x1': x1,
            'x2': x2,
            'y': y
        }

        print(f"save_dict: {save_dict}")

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        ckpt_name = ''
        if dtype == torch.bfloat16:
            ckpt_name = 'matmul_bf16'
        elif dtype == torch.float16:
            ckpt_name = 'matmul_fp16'

        torch.save(save_dict, f".tmp/{ckpt_name}.pt")

    def test_precision_ms(self, ckpt_name):
        # load from saved data
        # pt_data = torch.load(f".tmp/{ckpt_name}.pt", map_location="cpu")
        pt_data = torch.load(os.path.join(test_path, f"{ckpt_name}.pt"), map_location="cpu")

        x1_pt, x2_pt, y_pt = pt_data['x1'], pt_data['x2'], pt_data['y']

        # forward
        y_ms = x1_pt.matmul(x2_pt)

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        if np.allclose(y_ms.to(torch.float16), y_pt.to(torch.float16), 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


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
        TestAbs().test_api_completeness()
        TestAbsolute().test_api_completeness()
        TestAdd().test_api_completeness()
        TestCeil().test_api_completeness()
        TestClamp().test_api_completeness()
        TestClip().test_api_completeness()
        TestCos().test_api_completeness()
        TestDiv().test_api_completeness()
        TestDivide().test_api_completeness()
        TestExp().test_api_completeness()
        TestLog().test_api_completeness()
        TestSin().test_api_completeness()
        TestSquare().test_api_completeness()
        TestMul().test_api_completeness()
        TestDim().test_api_completeness()
        TestFloor().test_api_completeness()
        TestItem().test_api_completeness()
        TestNelement().test_api_completeness()
        TestNumel().test_api_completeness()
        TestSize().test_api_completeness()
        TestRound().test_api_completeness()
        TestSqrt().test_api_completeness()
        TestSub().test_api_completeness()
        TestTrunc().test_api_completeness()
        TestType().test_api_completeness()
        TestTypeas().test_api_completeness()
        TestMatmul().test_api_completeness()

    elif args.test_mode == 'outlier':
        TestAbs().test_api_outlier(args.backend)
        TestAbsolute().test_api_outlier(args.backend)
        TestAdd().test_api_outlier(args.backend)
        TestCeil().test_api_outlier(args.backend)
        TestClamp().test_api_outlier(args.backend)
        TestClip().test_api_outlier(args.backend)
        TestCos().test_api_outlier(args.backend)
        TestDiv().test_api_outlier(args.backend)
        TestDivide().test_api_outlier(args.backend)
        TestExp().test_api_outlier(args.backend)
        TestLog().test_api_outlier(args.backend)
        TestSin().test_api_outlier(args.backend)
        TestSquare().test_api_outlier(args.backend)
        TestMul().test_api_outlier(args.backend)
        TestDim().test_api_outlier(args.backend)
        TestFloor().test_api_outlier(args.backend)
        TestItem().test_api_outlier(args.backend)
        TestNelement().test_api_outlier(args.backend)
        TestNumel().test_api_outlier(args.backend)
        TestSize().test_api_outlier(args.backend)
        TestRound().test_api_outlier(args.backend)
        TestSqrt().test_api_outlier(args.backend)
        TestSub().test_api_outlier(args.backend)
        TestTrunc().test_api_outlier(args.backend)
        TestType().test_api_outlier(args.backend)
        TestTypeas().test_api_outlier(args.backend)
        TestMatmul().test_api_outlier(args.backend)

    elif args.test_mode == 'performance':
        TestAbs().test_performance(args.backend, 100, (10, 10))
        TestAbs().test_performance(args.backend, 100, (100, 100))
        TestAbs().test_performance(args.backend, 10, (1000, 100))
        TestAbs().test_performance(args.backend, 10, (1000, 1000))

        TestAbsolute().test_performance(args.backend, 100, (10, 10))
        TestAbsolute().test_performance(args.backend, 100, (100, 100))
        TestAbsolute().test_performance(args.backend, 10, (1000, 100))
        TestAbsolute().test_performance(args.backend, 10, (1000, 1000))

        TestAdd().test_performance(args.backend, 100, (10, 10))
        TestAdd().test_performance(args.backend, 100, (100, 100))
        TestAdd().test_performance(args.backend, 10, (1000, 100))
        TestAdd().test_performance(args.backend, 10, (1000, 1000))

        TestCeil().test_performance(args.backend, 100, (10, 10))
        TestCeil().test_performance(args.backend, 100, (100, 100))
        TestCeil().test_performance(args.backend, 10, (1000, 100))
        TestCeil().test_performance(args.backend, 10, (1000, 1000))

        TestClamp().test_performance(args.backend, 100, (10, 10))
        TestClamp().test_performance(args.backend, 100, (100, 100))
        TestClamp().test_performance(args.backend, 10, (1000, 100))
        TestClamp().test_performance(args.backend, 10, (1000, 1000))

        TestClip().test_performance(args.backend, 100, (10, 10))
        TestClip().test_performance(args.backend, 100, (100, 100))
        TestClip().test_performance(args.backend, 10, (1000, 100))
        TestClip().test_performance(args.backend, 10, (1000, 1000))

        TestCos().test_performance(args.backend, 100, (10, 10))
        TestCos().test_performance(args.backend, 100, (100, 100))
        TestCos().test_performance(args.backend, 10, (1000, 100))
        TestCos().test_performance(args.backend, 10, (1000, 1000))

        TestDiv().test_performance(args.backend, 100, (10, 10))
        TestDiv().test_performance(args.backend, 100, (100, 100))
        TestDiv().test_performance(args.backend, 10, (1000, 100))
        TestDiv().test_performance(args.backend, 10, (1000, 1000))

        TestDivide().test_performance(args.backend, 100, (10, 10))
        TestDivide().test_performance(args.backend, 100, (100, 100))
        TestDivide().test_performance(args.backend, 10, (1000, 100))
        TestDivide().test_performance(args.backend, 10, (1000, 1000))

        TestExp().test_performance(args.backend, 100, (10, 10))
        TestExp().test_performance(args.backend, 100, (100, 100))
        TestExp().test_performance(args.backend, 10, (1000, 100))
        TestExp().test_performance(args.backend, 10, (1000, 1000))

        TestLog().test_performance(args.backend, 100, (10, 10))
        TestLog().test_performance(args.backend, 100, (100, 100))
        TestLog().test_performance(args.backend, 10, (1000, 100))
        TestLog().test_performance(args.backend, 10, (1000, 1000))

        TestSin().test_performance(args.backend, 100, (10, 10))
        TestSin().test_performance(args.backend, 100, (100, 100))
        TestSin().test_performance(args.backend, 10, (1000, 100))
        TestSin().test_performance(args.backend, 10, (1000, 1000))

        TestSquare().test_performance(args.backend, 100, (10, 10))
        TestSquare().test_performance(args.backend, 100, (100, 100))
        TestSquare().test_performance(args.backend, 10, (1000, 100))
        TestSquare().test_performance(args.backend, 10, (1000, 1000))

        TestMul().test_performance(args.backend, 100, (10, 10))
        TestMul().test_performance(args.backend, 100, (100, 100))
        TestMul().test_performance(args.backend, 10, (1000, 100))
        TestMul().test_performance(args.backend, 10, (1000, 1000))

        TestDim().test_performance(args.backend, 100, (10, 10))
        TestDim().test_performance(args.backend, 100, (100, 100))
        TestDim().test_performance(args.backend, 10, (1000, 100))
        TestDim().test_performance(args.backend, 10, (1000, 1000))

        TestFloor().test_performance(args.backend, 100, (10, 10))
        TestFloor().test_performance(args.backend, 100, (100, 100))
        TestFloor().test_performance(args.backend, 10, (1000, 100))
        TestFloor().test_performance(args.backend, 10, (1000, 1000))

        TestItem().test_performance(args.backend, 100, (10, 10))
        TestItem().test_performance(args.backend, 100, (100, 100))
        TestItem().test_performance(args.backend, 10, (1000, 100))
        TestItem().test_performance(args.backend, 10, (1000, 1000))

        TestNelement().test_performance(args.backend, 100, (10, 10))
        TestNelement().test_performance(args.backend, 100, (100, 100))
        TestNelement().test_performance(args.backend, 10, (1000, 100))
        TestNelement().test_performance(args.backend, 10, (1000, 1000))

        TestNumel().test_performance(args.backend, 100, (10, 10))
        TestNumel().test_performance(args.backend, 100, (100, 100))
        TestNumel().test_performance(args.backend, 10, (1000, 100))
        TestNumel().test_performance(args.backend, 10, (1000, 1000))

        TestSize().test_performance(args.backend, 100, (10, 10))
        TestSize().test_performance(args.backend, 100, (100, 100))
        TestSize().test_performance(args.backend, 10, (1000, 100))
        TestSize().test_performance(args.backend, 10, (1000, 1000))

        TestRound().test_performance(args.backend, 100, (10, 10))
        TestRound().test_performance(args.backend, 100, (100, 100))
        TestRound().test_performance(args.backend, 10, (1000, 100))
        TestRound().test_performance(args.backend, 10, (1000, 1000))

        TestSqrt().test_performance(args.backend, 100, (10, 10))
        TestSqrt().test_performance(args.backend, 100, (100, 100))
        TestSqrt().test_performance(args.backend, 10, (1000, 100))
        TestSqrt().test_performance(args.backend, 10, (1000, 1000))

        TestSub().test_performance(args.backend, 100, (10, 10))
        TestSub().test_performance(args.backend, 100, (100, 100))
        TestSub().test_performance(args.backend, 10, (1000, 100))
        TestSub().test_performance(args.backend, 10, (1000, 1000))

        TestTrunc().test_performance(args.backend, 100, (10, 10))
        TestTrunc().test_performance(args.backend, 100, (100, 100))
        TestTrunc().test_performance(args.backend, 10, (1000, 100))
        TestTrunc().test_performance(args.backend, 10, (1000, 1000))

        TestType().test_performance(args.backend, 100, (10, 10))
        TestType().test_performance(args.backend, 100, (100, 100))
        TestType().test_performance(args.backend, 10, (1000, 100))
        TestType().test_performance(args.backend, 10, (1000, 1000))

        TestTypeas().test_performance(args.backend, 100, (10, 10))
        TestTypeas().test_performance(args.backend, 100, (100, 100))
        TestTypeas().test_performance(args.backend, 10, (1000, 100))
        TestTypeas().test_performance(args.backend, 10, (1000, 1000))

        TestMatmul().test_performance(args.backend, 100, (10, 10))
        TestMatmul().test_performance(args.backend, 100, (100, 100))
        TestMatmul().test_performance(args.backend, 10, (1000, 100))
        TestMatmul().test_performance(args.backend, 10, (1000, 1000))

    elif args.test_mode == 'precision':
        if args.backend == 'torch':
            TestAbs().test_precision_pt(torch.bfloat16)
            TestAbs().test_precision_pt(torch.float16)
            TestAbsolute().test_precision_pt(torch.bfloat16)
            TestAbsolute().test_precision_pt(torch.float16)
            TestAdd().test_precision_pt(torch.bfloat16)
            TestAdd().test_precision_pt(torch.float16)
            TestCeil().test_precision_pt(torch.bfloat16)
            TestCeil().test_precision_pt(torch.float16)
            TestClamp().test_precision_pt(torch.bfloat16)
            TestClamp().test_precision_pt(torch.float16)
            TestClip().test_precision_pt(torch.bfloat16)
            TestClip().test_precision_pt(torch.float16)
            TestCos().test_precision_pt(torch.bfloat16)
            TestCos().test_precision_pt(torch.float16)
            TestDiv().test_precision_pt(torch.bfloat16)
            TestDiv().test_precision_pt(torch.float16)
            TestDivide().test_precision_pt(torch.bfloat16)
            TestDivide().test_precision_pt(torch.float16)
            TestExp().test_precision_pt(torch.bfloat16)
            TestExp().test_precision_pt(torch.float16)
            TestLog().test_precision_pt(torch.bfloat16)
            TestLog().test_precision_pt(torch.float16)
            TestSin().test_precision_pt(torch.bfloat16)
            TestSin().test_precision_pt(torch.float16)
            TestSquare().test_precision_pt(torch.bfloat16)
            TestSquare().test_precision_pt(torch.float16)
            TestMul().test_precision_pt(torch.bfloat16)
            TestMul().test_precision_pt(torch.float16)
            TestFloor().test_precision_pt(torch.bfloat16)
            TestFloor().test_precision_pt(torch.float16)
            TestRound().test_precision_pt(torch.bfloat16)
            TestRound().test_precision_pt(torch.float16)
            TestSqrt().test_precision_pt(torch.bfloat16)
            TestSqrt().test_precision_pt(torch.float16)
            TestSub().test_precision_pt(torch.bfloat16)
            TestSub().test_precision_pt(torch.float16)
            TestTrunc().test_precision_pt(torch.bfloat16)
            TestTrunc().test_precision_pt(torch.float16)
            TestMatmul().test_precision_pt(torch.bfloat16)
            TestMatmul().test_precision_pt(torch.float16)

        elif args.backend == 'mindspore':
            TestAbs().test_precision_ms('abs_bf16')
            TestAbs().test_precision_ms('abs_fp16')
            TestAbsolute().test_precision_ms('absolute_bf16')
            TestAbsolute().test_precision_ms('absolute_fp16')
            TestAdd().test_precision_ms('add_bf16')
            TestAdd().test_precision_ms('add_fp16')
            # TestCeil().test_precision_ms('ceil_bf16')
            # TestCeil().test_precision_ms('ceil_fp16')
            TestClamp().test_precision_ms('clamp_bf16')
            TestClamp().test_precision_ms('clamp_fp16')
            TestClip().test_precision_ms('clip_bf16')
            TestClip().test_precision_ms('clip_fp16')
            TestCos().test_precision_ms('cos_bf16')
            TestCos().test_precision_ms('cos_fp16')
            TestDiv().test_precision_ms('div_bf16')
            TestDiv().test_precision_ms('div_fp16')
            TestDivide().test_precision_ms('divide_bf16')
            TestDivide().test_precision_ms('divide_fp16')
            TestExp().test_precision_ms('exp_bf16')
            TestExp().test_precision_ms('exp_fp16')
            TestLog().test_precision_ms('log_bf16')
            TestLog().test_precision_ms('log_fp16')
            TestSin().test_precision_ms('sin_bf16')
            TestSin().test_precision_ms('sin_fp16')
            TestSquare().test_precision_ms('square_bf16')
            TestSquare().test_precision_ms('square_fp16')
            TestMul().test_precision_ms('mul_bf16')
            TestMul().test_precision_ms('mul_fp16')
            TestFloor().test_precision_ms('floor_bf16')
            TestFloor().test_precision_ms('floor_fp16')
            # TestRound().test_precision_ms('round_bf16')
            # TestRound().test_precision_ms('round_fp16')
            TestSqrt().test_precision_ms('sqrt_bf16')
            TestSqrt().test_precision_ms('sqrt_fp16')
            TestSub().test_precision_ms('sub_bf16')
            TestSub().test_precision_ms('sub_fp16')
            TestTrunc().test_precision_ms('trunc_bf16')
            TestTrunc().test_precision_ms('trunc_fp16')
            TestMatmul().test_precision_ms('matmul_bf16')
            TestMatmul().test_precision_ms('matmul_fp16')