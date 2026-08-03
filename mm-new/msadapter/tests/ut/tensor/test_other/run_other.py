import os
import time
import random
import argparse

import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

import numpy as np

pt_input_output_path = "/home/workspace/mindspore_dataset/msadapter/test_input/ut/tensor/test_other"

dtype_str_dict = {
    torch.bfloat16: 'bfloat16',
    torch.float16: 'float16',
    torch.float32: 'float32',
}


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


class TestCumsum(TestCase):
    ops_name = "cumsum"

    def test_api_completeness(self):
        x = torch.randn(10)
        y = x.cumsum(0)

        x = torch.randn(2, 3)
        y = x.cumsum(1, dtype=torch.float32)

        x = torch.tensor([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]])
        y = x.cumsum(1, dtype=torch.bfloat16)

        print("cumsum: ", y)
        benchmark = torch.tensor([[3., 7., 13., 23.],
                                  [1., 7., 14., 23.],
                                  [4., 7., 15., 22.],
                                  [1., 4., 11., 20.]], dtype=torch.bfloat16)
        assert torch.equal(y, benchmark)

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(float('inf')),
            'nan': torch.empty(2, 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.cumsum(0)
            print('torch empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.cumsum(0)
            print('torch inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.cumsum(0)
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.cumsum(0)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.cumsum(0)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([2, 3]) and torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.cumsum(0)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([2, 3]) and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.2739,
            (100, 100): 1.2036,
            (1000, 100): 1.3846,
            (1000, 1000): 1.3262
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.3743,
            (100, 100): 0.5310,
            (1000, 100): 2.2761,
            (1000, 1000): 6.6517
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35653120.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.cumsum(0).npu()
                y = y.sum()
                print(y)

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
                y = x.cumsum(0)
                y = y.sum()
                print(y)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 5.25:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 5.25), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.26:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.26), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 0.34:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.34), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.cumsum(0)
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
        y_ms = x_pt.cumsum(0)
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


class TestDiag(TestCase):
    ops_name = "diag"

    def test_api_completeness(self):
        x = torch.arange(3)
        y = x.diag()
        print(y)
        y = x.diag(1)
        assert torch.equal(y, torch.tensor([[0, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 2],
                                            [0, 0, 0, 0]]))
        y = x.diag(diagonal=-1)
        print(y)

        x = torch.arange(9).reshape(3, 3)
        y = x.diag()
        print(y)
        y = x.diag(diagonal=1)
        print(y)
        y = x.diag(-1)
        assert torch.equal(y, torch.tensor([3, 7]))

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2).fill_(float('inf')),
            'nan': torch.empty(2).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.diag()
            print('torch empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.diag()
            print('torch inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.diag()
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.diag()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0, 0])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.diag()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([2]) and torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.diag()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([2]) and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.2299,
            (100, 100): 1.1622,
            (1000, 100): 1.3418,
            (1000, 1000): 1.3181
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.2660,
            (100, 100): 0.3212,
            (1000, 100): 1.6856,
            (1000, 1000): 2.1995
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35653120.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.diag().npu()
                y = y.sum()
                print(y)

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
            if shape in [(1000, 100)]:
                return
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()

                def flag_func():
                    pass

                y = x.diag().npu()
                y = y.sum()
                print(y)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 0.89:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.89), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 2.88:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 2.88), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 0.70:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.70), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.diag().npu()
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
        def flag_func():
            pass

        y_ms = x_pt.diag()
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


class TestFlatten(TestCase):
    ops_name = "flatten"

    def test_api_completeness(self):
        x = torch.randn(3 * 4 * 5).reshape(3, 4, 5)
        y = x.flatten()
        print(y)
        y = x.flatten(start_dim=0, end_dim=-1)
        print(y)
        y = x.flatten(1, 2)
        print(y)
        y = x.flatten(start_dim=1)
        print(y)
        y = x.flatten(end_dim=1)
        print(y)

        assert y.shape == torch.Size([12, 5])

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2 * 3).fill_(float('inf')),
            'nan': torch.empty(2 * 3).fill_(float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.flatten()
            print('torch empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.flatten()
            print('torch inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.flatten()
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.flatten()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.flatten()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([2 * 3]) and torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.flatten()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([2 * 3]) and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.1996,
            (100, 100): 1.2029,
            (1000, 100): 1.2042,
            (1000, 1000): 1.2613,
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.2586,
            (100, 100): 0.3168,
            (1000, 100): 1.6601,
            (1000, 1000): 2.2165,
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35653120.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.flatten().npu()
                y = y.sum()
                print(y)

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

                def flag_func():
                    pass

                y = x.flatten()
                y = y.sum()
                print(y)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 0.42:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.42), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.56:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.56), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 0.13:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.13), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.flatten()
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
        y_ms = x_pt.flatten()
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


class TestFill(TestCase):
    ops_name = "fill"

    def test_api_completeness(self):
        x = torch.empty(3 * 4 * 5).reshape(3, 4, 5)
        y = x.fill_(3.14)
        print(y)

        y = x.fill_(True)
        print(y)

        y = x.fill_(torch.tensor(2.7))
        print(y)

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.empty(2, 3).fill_(3.14),
            'nan': torch.empty(2, 3).fill_(3.14)
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.fill_(3.14)
            print('torch empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.fill_(3.14)
            print('torch inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.fill_(3.14)
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.fill_(3.14)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.fill_(3.14)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert torch.equal(y, pt_output['inf'])

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.fill_(3.14)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert torch.equal(y, pt_output['nan'])

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.1694,
            (100, 100): 1.1597,
            (1000, 100): 1.2129,
            (1000, 1000): 1.2052,
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35653120.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.fill_(3.14).npu()
                y = y.sum()
                print(y)

                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.fill_(3.14)
                y = y.sum()
                print(y)
                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(ms.runtime.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms '
                  f'({forward_cost_time / pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time / pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory / pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory / pt_max_memory[shape] * 1.2:.2f} std)'
                  )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 0.24:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.24), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 0.07:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.07), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        pass

    def test_precision_ms(self, dtype=torch.bfloat16):
        x = torch.empty(3 * 4 * 5, dtype=dtype).reshape(3, 4, 5)
        y = x.fill_(torch.tensor(2.7))
        print(y)

        benchmark = torch.ones_like(x, dtype=dtype) * 2.7
        assert torch.equal(y, benchmark)


class TestMaskedFill(TestCase):

    def test_api_completeness(self):
        x = torch.empty(3 * 4 * 5).reshape(3, 4, 5)
        mask = torch.randint_like(x, 0, 2, dtype=torch.bool)
        y = x.masked_fill_(mask, 3.14)
        print(y)

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.tensor([[3.1400, float('inf'), float('inf')],
                                 [float('inf'), float('inf'), 3.1400]]),
            'nan': torch.tensor([[3.1400, float('nan'), float('nan')],
                                 [float('nan'), float('nan'), 3.1400]])
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            mask = torch.Tensor().to(torch.bool).npu()
            y = x.masked_fill_(mask, 3.14)
            print('torch empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            mask = torch.zeros_like(x, dtype=torch.bool).npu()
            mask[0][0], mask[1][2] = True, True
            y = x.masked_fill_(mask, 3.14)
            print('torch inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            mask = torch.zeros_like(x, dtype=torch.bool).npu()
            mask[0][0], mask[1][2] = True, True
            y = x.masked_fill_(mask, 3.14)
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            mask = torch.Tensor([]).to(torch.bool).npu()
            y = x.masked_fill_(mask, 3.14)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

            x = torch.empty(2, 3).fill_(float('inf'))
            mask = torch.zeros_like(x, dtype=torch.bool).npu()
            mask[0][0], mask[1][2] = True, True
            y = x.masked_fill_(mask, 3.14)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])

            y_exclude_inf = y.where(~y.isinf(), 0.)
            pt_output_exclude_inf = pt_output['inf'].where(~y.isinf(), 0.)
            assert torch.equal(y_exclude_inf, pt_output_exclude_inf)

            x = torch.empty(2, 3).fill_(float('nan'))
            mask = torch.zeros_like(x, dtype=torch.bool).npu()
            mask[0][0], mask[1][2] = True, True
            y = x.masked_fill_(mask, 3.14)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])

            y_exclude_nan = y.where(~y.isnan(), 0.)
            pt_output_exclude_nan = pt_output['nan'].where(~y.isnan(), 0.)
            assert torch.equal(y_exclude_nan, pt_output_exclude_nan)

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.1190,
            (100, 100): 1.1262,
            (1000, 100): 1.2012,
            (1000, 1000): 1.1626,
        }

        pt_max_memory = {
            (10, 10): 33558016.0,
            (100, 100): 33587712.0,
            (1000, 100): 33857536.0,
            (1000, 1000): 36653568.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16).npu()
            mask = torch.randint_like(x, 0, 2, dtype=torch.bool)

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.masked_fill_(mask, 3.14).npu()
                y = y.sum()
                print(y)

                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16)
            mask = torch.randint_like(x, 0, 2, dtype=torch.bool)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.masked_fill_(mask, 3.14).npu()
                y = y.sum()
                print(y)
                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(ms.runtime.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms '
                  f'({forward_cost_time / pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time / pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory / pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory / pt_max_memory[shape] * 1.2:.2f} std)'
                  )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 0.37:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.37), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 0.10:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.10), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        pass

    def test_precision_ms(self, dtype=torch.bfloat16):
        x = torch.zeros(3 * 4 * 5, dtype=dtype).reshape(3, 4, 5)
        mask = torch.randint_like(x, 0, 2, dtype=torch.bool)
        y = x.masked_fill_(mask, 3.14)
        y_314 = y == 3.14
        y_0 = y == 0
        assert torch.equal(y_314, mask) and torch.equal(y_0, ~mask)


class TestNumpy(TestCase):
    ops_name = "numpy"

    def test_api_completeness(self):
        x = torch.randn(3 * 4 * 5).reshape(3, 4, 5)
        y = x.detach().cpu().numpy()

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': np.array([]),
            'inf': np.full((2, 3), fill_value=np.inf),
            'nan': np.full((2, 3), fill_value=np.nan)
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.detach().cpu().numpy()
            print('torch empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.detach().cpu().numpy()
            print('torch inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.detach().cpu().numpy()
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.detach().cpu().numpy()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == (0,)

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.detach().cpu().numpy()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == (2, 3) and np.all(np.isinf(y))

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.detach().cpu().numpy()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == (2, 3) and np.all(np.isnan(y))

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 0.1049,
            (100, 100): 0.1235,
            (1000, 100): 1.7735,
            (1000, 1000): 2.2810,
        }

        pt_max_memory = {
            (10, 10): 512.0,
            (100, 100): 20480.0,
            (1000, 100): 200192.0,
            (1000, 1000): 2000384.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.detach().cpu().to(torch.float32).numpy()

                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.detach().cpu().to(torch.float32).numpy()
                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(ms.runtime.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms '
                  f'({forward_cost_time / pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time / pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory / pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory / pt_max_memory[shape] * 1.2:.2f} std)'
                  )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 5.19:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 5.19), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 4.80:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 4.80), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()

        # forward and backward
        y = x.detach().cpu().to(torch.float32).numpy()

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y,
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt = pt_data['x'], pt_data['y']

        y_ms = x_pt.detach().cpu().to(torch.float32).numpy()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert np.array_equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"


class TestRepeat(TestCase):
    ops_name = "repeat"

    def test_api_completeness(self):
        x = torch.arange(4)
        y = x.repeat(3, 3)
        print(y)
        y = x.repeat(2)
        print(y)
        y = x.repeat(1, 2, 3)
        print(y)
        assert y.shape == torch.Size([1, 2, 12])

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.full((6, 6), float('inf')),
            'nan': torch.full((6, 6), float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.repeat(3, 2)
            print('torch empty: ', y)

            x = torch.full((2, 3), float('inf')).npu()
            y = x.repeat(3, 2)
            print('torch inf: ', y)

            x = torch.full((2, 3), float('nan')).npu()
            y = x.repeat(3, 2)
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.repeat(3, 2)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([3, 0])

            x = torch.full((2, 3), float('inf'))
            y = x.repeat(3, 2)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([6, 6]) and torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.repeat(3, 2)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([6, 6]) and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.1904,
            (100, 100): 1.1915,
            (1000, 100): 1.2582,
            (1000, 1000): 1.2888,
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.2548,
            (100, 100): 0.3169,
            (1000, 100): 1.6766,
            (1000, 1000): 2.1885,
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35653120.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.repeat(3, 2).npu()
                y = y.sum()
                print(y)

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

                def flag_func():
                    pass

                y = x.repeat(3, 2)
                y = y.sum()
                print(y)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 0.48:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.48), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.82:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.82), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 0.54:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.54), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.repeat(3, 2)
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
        y_ms = x_pt.repeat(3, 2)
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


class TestRepeatInterleave(TestCase):
    ops_name = "repeat_interleave"

    def test_api_completeness(self):

        x = torch.arange(4).reshape(2, 2)

        y = x.repeat_interleave(2)
        print(y)

        y = x.repeat_interleave(2)
        print(y)

        y = x.repeat_interleave(3, dim=1)
        print(y)

        y = x.repeat_interleave(torch.tensor([1, 2]), dim=0)
        print(y)

        y = x.repeat_interleave(torch.tensor([1, 2]), dim=0, output_size=3)
        print(y)

        assert torch.equal(y, torch.tensor([[0, 1],
                                            [2, 3],
                                            [2, 3]]))

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.full((12,), float('inf')),
            'nan': torch.full((12,), float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.repeat_interleave(2)
            print('torch empty: ', y)

            x = torch.full((2, 3), float('inf')).npu()
            y = x.repeat_interleave(2)
            print('torch inf: ', y)

            x = torch.full((2, 3), float('nan')).npu()
            y = x.repeat_interleave(2)
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.repeat_interleave(2)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

            x = torch.full((2, 3), float('inf'))
            y = x.repeat_interleave(2)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([12]) and torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.repeat_interleave(2)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([12]) and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []
        repeat_times_dict = {
            (10, 10): 50,
            (100, 100): 20,
            (1000, 100): 20,
            (1000, 1000): 10,
        }

        repeats_dict = {
            (10, 10): 40000,
            (100, 100): 400,
            (1000, 100): 40,
            (1000, 1000): 4,
        }

        pt_forward_time_diff_shape = {
            (10, 10): 2.0746,
            (100, 100): 8.8147,
            (1000, 100): 87.9719,
            (1000, 1000): 818.2020,
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.2771,
            (100, 100): 0.3388,
            (1000, 100): 1.6681,
            (1000, 1000): 2.3281,
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35557376.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times_dict[shape]):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.repeat_interleave(repeats=repeats_dict[shape])
                y = y.sum()
                print(y)

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

            for _ in range(repeat_times_dict[shape]):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()

                y = x.repeat_interleave(repeats=repeats_dict[shape])
                y = y.sum()
                print(y)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 8.71:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 8.71), but got '
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
            if max_memory > pt_max_memory[shape] * 0.97:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.97), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.repeat_interleave(100)
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
        y_ms = x_pt.repeat_interleave(100)
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


class TestTolist(TestCase):
    ops_name = "tolist"

    def test_api_completeness(self):
        x = torch.randn(3 * 4 * 5).reshape(3, 4, 5)
        y = x.tolist()
        print(y)

    def test_api_outlier(self, backend):
        inf, nan = float('inf'), float('nan')
        pt_output = {
            'empty': [],
            'inf': [[inf, inf, inf], [inf, inf, inf]],
            'nan': [[nan, nan, nan], [nan, nan, nan]]
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.tolist()
            print('torch empty: ', y)

            x = torch.empty(2, 3).fill_(inf).npu()
            y = x.tolist()
            print('torch inf: ', y)

            x = torch.empty(2, 3).fill_(nan).npu()
            y = x.tolist()
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.tolist()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y == pt_output['empty']

            x = torch.empty(2, 3).fill_(inf)
            y = x.tolist()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            y = np.array(y)
            assert y.shape == (2, 3) and np.isinf(y).all()

            x = torch.empty(2, 3).fill_(nan)
            y = x.tolist()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            y = np.array(y)
            assert y.shape == (2, 3) and np.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        repeat_times_dict = {
            (10, 10): 50,
            (100, 100): 20,
            (1000, 100): 10,
            (1000, 1000): 10,
        }

        pt_forward_time_diff_shape = {
            (10, 10): 0.0719,
            (100, 100): 0.4294,
            (1000, 100): 8.0867,
            (1000, 1000): 100.3250,
        }

        pt_max_memory = {
            (10, 10): 512.0,
            (100, 100): 20480.0,
            (1000, 100): 200192.0,
            (1000, 1000): 2000384.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16).npu()

            for _ in range(repeat_times_dict[shape]):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.tolist()

                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            x = torch.randn(shape, dtype=torch.bfloat16)

            for _ in range(repeat_times_dict[shape]):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.tolist()
                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(ms.runtime.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms '
                  f'({forward_cost_time / pt_forward_time_diff_shape[shape]:.3f} pta) '
                  f'({forward_cost_time / pt_forward_time_diff_shape[shape] * 1.2:.2f} std), '
                  f'max memory {max_memory} Byte '
                  f'({max_memory / pt_max_memory[shape]:.2f} pta) '
                  f'({max_memory / pt_max_memory[shape] * 1.2:.2f} std)'
                  )

            # check forward performence
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 24.40:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 24.40), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()

        # forward and backward
        y = x.tolist()

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y,
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt = pt_data['x'], pt_data['y']

        y_ms = x_pt.tolist()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert np.array_equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"


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
    TestList = [TestCumsum, TestDiag, TestFlatten, TestFill, TestMaskedFill, TestNumpy, TestRepeat,
                TestRepeatInterleave, TestTolist]
    # TestList = [TestTolist]
    test_precision_dtype = [torch.bfloat16, torch.float16]

    for TestCase in TestList:
        print(f"running test {TestCase.__name__} {args.test_mode}")
        test_case = TestCase()
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
