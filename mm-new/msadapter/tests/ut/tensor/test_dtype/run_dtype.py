import os
import time
import random
import argparse
import unittest

import torch
import torch_npu


import numpy as np

pt_input_output_path = "/home/workspace/mindspore_dataset/msadapter/test_input/ut/tensor/test_dtype"

dtype_str_dict = {
    torch.bfloat16: 'bfloat16',
    torch.float16: 'float16',
    torch.float32: 'float32',
}

inf, nan = float('inf'), float('nan')

def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


class TestFloat(unittest.TestCase):
    ops_name = "float"
    def test_api_completeness(self):
        x = torch.randn(11, dtype=torch.float16)
        y = x.float()
        assert y.dtype == torch.float32

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': (torch.tensor([]), torch.tensor([])),
            'inf': (torch.full((1, 3), inf), torch.full((1, 3), inf)),
            'nan': (torch.full((1, 3), nan), torch.full((1, 3), nan))
        }

        if backend == 'torch':
            x = torch.Tensor().half().npu()
            y = x.float()
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).half().npu()
            y = x.float()
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).half().npu()
            y = x.float()
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).half().npu()
            y = x.float()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.dtype == torch.float32 and y.shape == torch.Size([0])

            x = torch.full((2, 3), inf).half().npu()
            y = x.float()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.dtype == torch.float32 and torch.isinf(y).all()

            x = torch.full((2, 3), nan).half().npu()
            y = x.float()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.dtype == torch.float32 and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0745,
            (100, 100):   0.0747,
            (1000, 100):  0.0919,
            (1000, 1000): 0.0960,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2138,
            (100, 100):   0.2768,
            (1000, 100):  2.3070,
            (1000, 1000): 2.9629,
        }

        pt_max_memory = {
            (10, 10):     3072.0,
            (100, 100):   82944.0,
            (1000, 100):  802304.0,
            (1000, 1000): 8002560.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.float()
                y = y.sum()

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
            ms.runtime.launch_blocking()

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.float()
                y = y.sum()
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
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'backward {backward_cost_time:.4f} ms '
                  f'({backward_cost_time/pt_backward_time_diff_shape[shape]:.3f} pta) '
                  f'({backward_cost_time/pt_backward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory/pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory/pt_max_memory[shape] * 1.2:.2f} std)'
            )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 4.49:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 4.49), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.94:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.94), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []
        def x_hook(grad):
            x_grad.append(grad)
        x = torch.randn((100, 100), requires_grad=True).to(dtype).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.float()
        y = y.sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt, x_grad_pt = pt_data['x'].npu(), pt_data['y'].npu(), pt_data['x_grad'].npu()

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.float()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"

         # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        assert np.allclose(weight_grad[0].to(torch.float32).cpu(), x_grad_pt.to(torch.float32).cpu(), 0.0, 0.0)


class TestHalf(unittest.TestCase):
    ops_name = "half"
    def test_api_completeness(self):
        x = torch.randn(11, dtype=torch.float32)
        y = x.half()
        assert y.dtype == torch.float16

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': (torch.tensor([]), torch.tensor([])),
            'inf': (torch.full((1, 3), inf), torch.full((1, 3), inf)),
            'nan': (torch.full((1, 3), nan), torch.full((1, 3), nan))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.half()
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            y = x.half()
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            y = x.half()
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.half()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.dtype == torch.float16 and y.shape == torch.Size([0])

            x = torch.full((2, 3), inf).npu()
            y = x.half()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.dtype == torch.float16 and torch.isinf(y).all()

            x = torch.full((2, 3), nan).npu()
            y = x.half()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.dtype == torch.float16 and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0744,
            (100, 100):   0.0773,
            (1000, 100):  0.0944,
            (1000, 1000): 0.0971,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2104,
            (100, 100):   0.2336,
            (1000, 100):  1.8211,
            (1000, 1000): 3.2413,
        }

        pt_max_memory = {
            (10, 10):     3072.0,
            (100, 100):   102912.0,
            (1000, 100):  1002496.0,
            (1000, 1000): 10002432.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.float32, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.half()
                y = y.sum()

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
            ms.runtime.launch_blocking()

            x = torch.randn(shape, dtype=torch.float32, requires_grad=True)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.half()
                y = y.sum()
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
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'backward {backward_cost_time:.4f} ms '
                  f'({backward_cost_time/pt_backward_time_diff_shape[shape]:.3f} pta) '
                  f'({backward_cost_time/pt_backward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory/pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory/pt_max_memory[shape] * 1.2:.2f} std)'
            )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 4.68:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 4.68), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 2.05:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 2.05), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []
        def x_hook(grad):
            x_grad.append(grad)
        x = torch.randn((100, 100), requires_grad=True).to(dtype).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.half()
        y = y.sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.half()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()
        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"

         # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        assert np.allclose(weight_grad[0].to(torch.float32), x_grad_pt.to(torch.float32), 0.0, 0.0)


class TestBfloat16(unittest.TestCase):
    ops_name = "bfloat16"
    def test_api_completeness(self):
        x = torch.randn(11, dtype=torch.float32)
        y = x.bfloat16()
        assert y.dtype == torch.bfloat16

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': (torch.tensor([]), torch.tensor([])),
            'inf': (torch.full((1, 3), inf), torch.full((1, 3), inf)),
            'nan': (torch.full((1, 3), nan), torch.full((1, 3), nan))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.bfloat16()
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            y = x.bfloat16()
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            y = x.bfloat16()
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.bfloat16()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.dtype == torch.bfloat16 and y.shape == torch.Size([0])

            x = torch.full((2, 3), inf).npu()
            y = x.bfloat16()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.dtype == torch.bfloat16 and torch.isinf(y).all()

            x = torch.full((2, 3), nan).npu()
            y = x.bfloat16()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.dtype == torch.bfloat16 and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0719,
            (100, 100):   0.0738,
            (1000, 100):  0.0868,
            (1000, 1000): 0.0949,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2141,
            (100, 100):   0.2453,
            (1000, 100):  1.7548,
            (1000, 1000): 3.2303,
        }

        pt_max_memory = {
            (10, 10):     3072.0,
            (100, 100):   102912.0,
            (1000, 100):  1002496.0,
            (1000, 1000): 10002432.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.float32, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.bfloat16()
                y = y.sum()

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
            ms.runtime.launch_blocking()

            x = torch.randn(shape, dtype=torch.float32, requires_grad=True)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.bfloat16()
                y = y.sum()
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
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'backward {backward_cost_time:.4f} ms '
                  f'({backward_cost_time/pt_backward_time_diff_shape[shape]:.3f} pta) '
                  f'({backward_cost_time/pt_backward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory/pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory/pt_max_memory[shape] * 1.2:.2f} std)'
            )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 4.68:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 4.68), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.98:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.98), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []
        def x_hook(grad):
            x_grad.append(grad)
        x = torch.randn((100, 100), requires_grad=True).to(dtype).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.bfloat16()
        y = y.sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.bfloat16()
        y_ms = y_ms.sum()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"

         # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        assert np.allclose(weight_grad[0].to(torch.float32), x_grad_pt.to(torch.float32), 0.0, 0.0)


class TestInt(unittest.TestCase):
    ops_name = "int"
    def test_api_completeness(self):
        x = torch.randint(0, 1024, (3, 2))
        y = x.int()
        assert y.dtype == torch.int32

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]).int(),
            'inf': torch.full((2, 3), 2147483647, dtype=torch.int32).npu(),
            'nan': torch.full((2, 3), 0, dtype=torch.int32).npu(),
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.int()
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            y = x.int()
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            y = x.int()
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.int()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.dtype == torch.int32 and y.shape == torch.Size([0])

            x = torch.full((2, 3), inf).npu()
            y = x.int()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.dtype == torch.int32 and torch.equal(y, pt_output['inf'])

            x = torch.full((2, 3), nan).npu()
            y = x.int()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.dtype == torch.int32 and torch.equal(y, pt_output['nan'])

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0752,
            (100, 100):   0.0469,
            (1000, 100):  0.0536,
            (1000, 1000): 0.0565,
        }

        pt_max_memory = {
            (10, 10):     4096.0,
            (100, 100):   163328.0,
            (1000, 100):  1614336.0,
            (1000, 1000): 16014336.0,
        }

        if backend == 'torch':
            x = torch.empty(shape, dtype=torch.float32).npu()
            x.uniform_(0., 1024.)
            x.requires_grad_ = True
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.int()
                y = y.sum()

                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            ms.runtime.launch_blocking()

            x = torch.empty(shape, dtype=torch.float32, requires_grad=True).npu()
            x.uniform_(0., 1024.)
            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.int()
                y = y.sum()
                forward_cost_time.append(time.time() - forward_start_time)

                max_memory.append(ms.runtime.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory/pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory/pt_max_memory[shape] * 1.2:.2f} std)'
            )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 1.31:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 1.31), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):

        x = torch.empty((100, 100)).npu()
        x.uniform_(0., 1024.)
        x = x.to(dtype)

        # forward and backward
        y = x.int()
        y = y.sum()
        print(f"output is {y}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt = pt_data['x'], pt_data['y']

        y_ms = x_pt.int()
        y_ms = y_ms.sum()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"


class TestLong(unittest.TestCase):
    ops_name = "long"
    def test_api_completeness(self):
        x = torch.randint(0, 1024, (3, 2))
        y = x.long()
        assert y.dtype == torch.int64

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]).long(),
            'inf': torch.full((2, 3), 9223372036854775807, dtype=torch.int64).npu(),
            'nan': torch.full((2, 3), 0, dtype=torch.int64).npu(),
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.long()
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            y = x.long()
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            y = x.long()
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.long()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.dtype == torch.int64 and y.shape == torch.Size([0])

            x = torch.full((2, 3), inf).npu()
            y = x.long()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.dtype == torch.int64 and torch.equal(y, pt_output['inf'])

            x = torch.full((2, 3), nan).npu()
            y = x.long()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.dtype == torch.int64 and torch.equal(y, pt_output['nan'])

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0739,
            (100, 100):   0.0513,
            (1000, 100):  0.0543,
            (1000, 1000): 0.0563,
        }

        pt_max_memory = {
            (10, 10):     3584.0,
            (100, 100):   122880.0,
            (1000, 100):  1213952.0,
            (1000, 1000): 12014080.0,
        }

        if backend == 'torch':
            x = torch.empty(shape, dtype=torch.float32).npu()
            x.uniform_(0., 1024.)
            x.requires_grad_ = True
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.long()
                y = y.sum()

                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            ms.runtime.launch_blocking()

            x = torch.empty(shape, dtype=torch.float32, requires_grad=True).npu()
            x.uniform_(0., 1024.)
            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.long()
                y = y.sum()
                forward_cost_time.append(time.time() - forward_start_time)

                max_memory.append(ms.runtime.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory/pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory/pt_max_memory[shape] * 1.2:.2f} std)'
            )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 1.34:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 1.34), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):

        x = torch.empty((100, 100)).npu()
        x.uniform_(0., 1024.)
        x = x.to(dtype)

        # forward and backward
        y = x.long()
        y = y.sum()
        print(f"output is {y}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt = pt_data['x'], pt_data['y']

        y_ms = x_pt.long()
        y_ms = y_ms.sum()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"


class TestByte(unittest.TestCase):
    ops_name = "byte"
    def test_api_completeness(self):
        x = torch.randint(0, 256, (3, 2))
        y = x.byte()
        assert y.dtype == torch.uint8

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]).byte(),
            'inf': torch.full((2, 3), 255, dtype=torch.uint8).npu(),
            'nan': torch.full((2, 3), 0, dtype=torch.uint8).npu(),
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.byte()
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            y = x.byte()
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            y = x.byte()
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.byte()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.dtype == torch.uint8 and y.shape == torch.Size([0])

            x = torch.full((2, 3), inf).npu()
            y = x.byte()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.dtype == torch.uint8 and torch.equal(y, pt_output['inf'])

            x = torch.full((2, 3), nan).npu()
            y = x.byte()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.dtype == torch.uint8 and torch.equal(y, pt_output['nan'])

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0756,
            (100, 100):   0.0486,
            (1000, 100):  0.0515,
            (1000, 1000): 0.0532,
        }

        pt_max_memory = {
            (10, 10):     4096.0,
            (100, 100):   133120.0,
            (1000, 100):  1314304.0,
            (1000, 1000): 13014528.0,
        }

        if backend == 'torch':
            x = torch.empty(shape, dtype=torch.float32).npu()
            x.uniform_(0., 255.)
            x.requires_grad_ = True
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.byte()
                y = y.sum()

                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            ms.runtime.launch_blocking()

            x = torch.empty(shape, dtype=torch.float32, requires_grad=True).npu()
            x.uniform_(0., 1024.)
            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.byte()
                y = y.sum()
                forward_cost_time.append(time.time() - forward_start_time)

                max_memory.append(ms.runtime.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory/pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory/pt_max_memory[shape] * 1.2:.2f} std)'
            )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 1.26:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 1.26), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):

        x = torch.empty((100, 100)).npu()
        x.uniform_(0., 255.)
        x = x.to(dtype)

        # forward and backward
        y = x.byte()
        y = y.sum()
        print(f"output is {y}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt = pt_data['x'], pt_data['y']

        y_ms = x_pt.byte()
        y_ms = y_ms.sum()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"


class TestBool(unittest.TestCase):
    ops_name = "bool"
    def test_api_completeness(self):
        x = torch.randint(0, 2, (3, 2))
        y = x.bool()
        assert y.dtype == torch.bool

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]).bool(),
            'inf': torch.full((2, 3), True, dtype=torch.bool).npu(),
            'nan': torch.full((2, 3), True, dtype=torch.bool).npu(),
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.bool()
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            y = x.bool()
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            y = x.bool()
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.bool()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.dtype == torch.bool and y.shape == torch.Size([0])

            x = torch.full((2, 3), inf).npu()
            y = x.bool()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.dtype == torch.bool and torch.equal(y, pt_output['inf'])

            x = torch.full((2, 3), nan).npu()
            y = x.bool()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.dtype == torch.bool and torch.equal(y, pt_output['nan'])

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.1090,
            (100, 100):   0.0518,
            (1000, 100):  0.0551,
            (1000, 1000): 0.0571,
        }

        pt_max_memory = {
            (10, 10):     4096.0,
            (100, 100):   86016.0,
            (1000, 100):  715776.0,
            (1000, 1000): 7015936.0,
        }

        if backend == 'torch':
            x = torch.randint(0, 2, shape, dtype=torch.bfloat16).npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.bool()
                y = y.sum()

                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            ms.runtime.launch_blocking()

            x = torch.randint(0, 2, shape, dtype=torch.bfloat16).npu()
            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.bool()
                y = y.sum()
                forward_cost_time.append(time.time() - forward_start_time)

                max_memory.append(ms.runtime.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time/pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory/pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory/pt_max_memory[shape] * 1.2:.2f} std)'
            )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 1.25:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 1.25), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        shape = (100, 100)
        x = torch.randint(0, 2, shape).to(dtype).npu()

        # forward and backward
        y = x.bool()
        y = y.sum()
        print(f"output is {y}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt = pt_data['x'], pt_data['y']

        y_ms = x_pt.bool()
        y_ms = y_ms.sum()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance', 'precision', 'outlier'],
                        help="test mode")
    parser.add_argument('--backend', type=str, choices=['torch', 'mindspore'],
                        help="backend")
    args, _ = parser.parse_known_args()

    test_shape = [(10, 10), (100, 100), (1000, 100), (1000, 1000)]
    repeat_times = [50, 50, 50, 50]

    TestList = [TestFloat, TestHalf, TestBfloat16, TestInt, TestLong, TestByte, TestBool]
    test_precision_dtype = [torch.bfloat16, torch.float16, torch.float32]

    for unittest.TestCase in TestList:
        print(f"running test {unittest.TestCase.__name__} {args.test_mode}")
        test_case = unittest.TestCase()
        if args.test_mode == 'completeness':
            test_case.test_api_completeness()
        elif args.test_mode == 'performance':
            for i, shape in enumerate(test_shape):
                print(f"running test shape {shape}")
                test_case.test_performance(args.backend, repeat_times[i], test_shape[i])
                print(f"test shape {shape} success")
        elif args.test_mode == 'precision':
            for dtype in test_precision_dtype:
                print(f"running {args.backend} test dtype {dtype}")
                if args.backend == 'torch':
                    test_case.test_precision_pt(dtype)
                elif args.backend == 'mindspore':
                    test_case.test_precision_ms(dtype)
                print(f"{args.backend} test dtype {dtype} success")
        elif args.test_mode == 'outlier':
            test_case.test_api_outlier(args.backend)
