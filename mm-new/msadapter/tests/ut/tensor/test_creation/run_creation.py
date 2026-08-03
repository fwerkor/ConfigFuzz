import os
import time
import random
import unittest
import argparse

import torch
import torch_npu


import numpy as np

pt_input_output_path = "/home/workspace/mindspore_dataset/msadapter/test_input/ut/tensor/test_creation"

inf, nan = float('inf'), float('nan')

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

class TestUniform_(unittest.TestCase):
    ops_name = "uniform"
    def test_api_completeness(self):
        x = torch.empty(5)

        x.uniform_()
        self.assertTrue(x.min() >= 0 and x.max() < 1)

        x.uniform_(2, to=3)
        self.assertTrue(x.min() >= 2 and x.max() < 3)
        self.assertTrue(torch.allclose(x.mean(), torch.tensor(2.5), 0.1, 0.1))


    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            x.uniform_()
            print('torch empty: ', x)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            x.uniform_()
            print('ms empty: ', x, ' torch empty: ', pt_output['empty'])
            assert x.shape == torch.Size([0])


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0688,
            (100, 100):   0.0666,
            (1000, 100):  0.0774,
            (1000, 1000): 0.0766,
        }

        pt_max_memory = {
            (10, 10):     2048.0,
            (100, 100):   22016.0,
            (1000, 100):  201728.0,
            (1000, 1000): 2001920.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                x.uniform_()
                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            ms.context.set_context(pynative_synchronize=True)

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                x.uniform_()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 3.4: # runtime problem
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.34), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        pass

    def test_precision_ms(self, dtype=torch.bfloat16):
        pass


class TestZero_(unittest.TestCase):
    ops_name = "zero"
    def test_api_completeness(self):
        x = torch.empty(5)
        x.zero_()
        self.assertTrue((x == 0.).all())

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.zero_()
            print('torch empty: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.zero_()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0281,
            (100, 100):   0.0250,
            (1000, 100):  0.0280,
            (1000, 1000): 0.0283,
        }

        pt_max_memory = {
            (10, 10):     512.0,
            (100, 100):   20480.0,
            (1000, 100):  200192.0,
            (1000, 1000): 2000384.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                x.zero_()
                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            ms.context.set_context(pynative_synchronize=True)

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                x.zero_()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 6.4: # runtime problem
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.64), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        pass

    def test_precision_ms(self, dtype=torch.bfloat16):
        pass


class TestCopy_(unittest.TestCase):
    ops_name = "copy"
    def test_api_completeness(self):
        x = torch.empty(3, 3)
        src = torch.randn_like(x)
        x.copy_(src)
        self.assertTrue(torch.equal(src, x))
        x.copy_(src, non_blocking=True)


    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            src = torch.randn_like(x)
            x.copy_(src)
            print('torch empty: ', x)

            x = torch.full((2, 3), float('inf')).npu()
            src = torch.randn_like(x)
            x.copy_(src)
            print('torch inf: ', x)

            x = torch.full((2, 3), float('nan')).npu()
            src = torch.randn_like(x)
            x.copy_(src)
            print('torch nan: ', x)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            src = torch.randn_like(x)
            x.copy_(src)
            print('ms empty: ', x, ' torch empty: ', pt_output['empty'])
            assert x.shape == torch.Size([0])

            x = torch.full((2, 3), float('inf'))
            src = torch.randn_like(x)
            x.copy_(src)
            print('ms inf: ', x)
            self.assertTrue(torch.equal(src, x))

            x = torch.empty(2, 3).fill_(float('nan'))
            src = torch.randn_like(x)
            x.copy_(src)
            print('ms nan: ', x)
            self.assertTrue(torch.equal(src, x))

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0627,
            (100, 100):   0.0603,
            (1000, 100):  0.0599,
            (1000, 1000): 0.0326,
        }

        pt_max_memory = {
            (10, 10):     1024.0,
            (100, 100):   40960.0,
            (1000, 100):  400384.0,
            (1000, 1000): 4000768.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            src = torch.randn_like(x)

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                x.copy_(src)
                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            ms.context.set_context(pynative_synchronize=True)

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            src = torch.randn_like(x)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                x.copy_(src)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 3.8: # runtime problem
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.38), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.20:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.20), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        pass

    def test_precision_ms(self, dtype=torch.bfloat16):
        pass


class TestExponential_(unittest.TestCase):
    ops_name = "exponential"
    def test_api_completeness(self):
        x = torch.empty(5)

        x.exponential_()
        self.assertTrue((x >= 0).all())
        self.assertFalse(x.isnan().any())
        self.assertFalse(x.isinf().any())

        x.exponential_(lambd=2, generator=None)

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            x.exponential_()
            print('torch empty: ', x)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            x.exponential_()
            print('ms empty: ', x, ' torch empty: ', pt_output['empty'])
            assert x.shape == torch.Size([0])


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.1155,
            (100, 100):   0.1093,
            (1000, 100):  0.1201,
            (1000, 1000): 0.1208,
        }

        pt_max_memory = {
            (10, 10):     2048.0,
            (100, 100):   40960.0,
            (1000, 100):  400384.0,
            (1000, 1000): 4000768.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                x.exponential_()
                forward_cost_time.append(time.time() - forward_start_time)
                max_memory.append(torch.npu.max_memory_allocated())

            max_memory = np.array(max_memory)[1:].mean()
            forward_cost_time = (np.array(forward_cost_time)[1:] * 1000).mean()

            print(f'{self.__class__.__name__:<10} {str(shape):<12} '
                  f'single forward {forward_cost_time:.4f} ms, '
                  f'max memory {max_memory} Byte')

        elif backend == 'mindspore':
            import mindspore as ms
            ms.context.set_context(pynative_synchronize=True)

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                x.exponential_()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 306.42:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 306.42), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 3.60:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 3.60), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        pass

    def test_precision_ms(self, dtype=torch.bfloat16):
        pass


class TestExpandAs(unittest.TestCase):
    ops_name = "expand_as"

    def test_api_completeness(self):
        x = torch.arange(4)

        other = torch.empty(1, 4)
        x = x.expand_as(other)
        self.assertTrue(x.shape == torch.Size([1, 4]))

        other = torch.empty(3, 4)
        x = x.expand_as(other=other)
        self.assertTrue(x.shape == torch.Size([3, 4]))

    def test_api_outlier(self, backend):

        pt_output = {
            'inf': torch.full((2, 3), float('inf')),
            'nan': torch.full((2, 3), float('nan'))
        }

        if backend == 'torch':

            x = torch.full((1, 3), float('inf')).npu()
            other = torch.empty(2, 3)
            y = x.expand_as(other)
            print('torch inf: ', y)

            x = torch.full((1, 3), float('nan')).npu()
            other = torch.empty(2, 3)
            y = x.expand_as(other)
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.full((1, 3), float('inf'))
            other = torch.empty(2, 3)
            y = x.expand_as(other)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([2, 3]) and torch.isinf(y).all()

            x = torch.empty(1, 3).fill_(float('nan'))
            other = torch.empty(2, 3)
            y = x.expand_as(other)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([2, 3]) and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0967,
            (100, 100):   0.0963,
            (1000, 100):  0.1091,
            (1000, 1000): 0.1148,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2356,
            (100, 100):   0.2969,
            (1000, 100):  1.5980,
            (1000, 1000): 2.1163,
        }

        pt_max_memory = {
            (10, 10):     4096.0,
            (100, 100):   83968.0,
            (1000, 100):  803328.0,
            (1000, 1000): 8003584.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            other = torch.empty((2, shape[0], shape[1]))

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.expand_as(other).npu()
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
            ms.context.set_context(pynative_synchronize=True)

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
            other = torch.empty((2, shape[0], shape[1]))

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.expand_as(other)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 34.9: # runtime problem
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 3.49), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 2.08:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 2.08), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 2.40:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 2.40), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')


    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []
        def x_hook(grad):
            x_grad.append(grad)
        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        other = torch.empty((2, x.shape[0], x.shape[1]))
        x.register_hook(x_hook)

        # forward and backward
        y = x.expand_as(other)
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
        other = torch.empty((2, x_pt.shape[0], x_pt.shape[1]))

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.expand_as(other)
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
    TestList = [TestUniform_, TestZero_, TestCopy_, TestExponential_, TestExpandAs]
    # TestList = [TestExpandAs]
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
