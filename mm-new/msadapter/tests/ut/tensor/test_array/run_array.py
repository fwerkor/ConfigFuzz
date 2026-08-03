import os
import time
import random
import argparse
import unittest

import torch
import torch_npu


import numpy as np

pt_input_output_path = "/home/workspace/mindspore_dataset/msadapter/test_input/ut/tensor/test_array"

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


class TestChunk(unittest.TestCase):
    ops_name = "chunk"
    def test_api_completeness(self):
        x = torch.arange(11)
        y = x.chunk(6)
        print(y)

        x = torch.arange(12).view((3, 4))
        y = x.chunk(2, 1)
        print(y)

        assert len(y) == 2 and y[0].shape == (3, 2) and y[1].shape == (3, 2)

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': (torch.tensor([]), torch.tensor([])),
            'inf': (torch.full((1, 3), inf), torch.full((1, 3), inf)),
            'nan': (torch.full((1, 3), nan), torch.full((1, 3), nan))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.chunk(2)
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            y = x.chunk(2)
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            y = x.chunk(2)
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.chunk(2)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            # assert isinstance(y, tuple) and y[0].shape == torch.Size([0]) and y[1].shape == torch.Size([0])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.chunk(2)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert isinstance(y, tuple) and torch.isinf(y[0]).all() and torch.isinf(y[1]).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.chunk(2)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert isinstance(y, tuple) and torch.isnan(y[0]).all() and torch.isnan(y[1]).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0728,
            (100, 100):   0.0705,
            (1000, 100):  0.0871,
            (1000, 1000): 0.0902,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2313,
            (100, 100):   0.2832,
            (1000, 100):  1.7552,
            (1000, 1000): 2.9653,
        }

        pt_max_memory = {
            (10, 10):     3584.0,
            (100, 100):   62976.0,
            (1000, 100):  602624.0,
            (1000, 1000): 6003200.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.chunk(2)
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
            ms.runtime.launch_blocking()

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.chunk(2)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 5.42:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 5.42), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 2.06:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 2.06), but got '
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
        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.chunk(2)
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
        y_ms = x_pt.chunk(2)
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


class TestGather(unittest.TestCase):
    ops_name = "gather"
    def test_api_completeness(self):
        x = torch.randn(3, 4)
        dim = 0
        index = torch.tensor([[0, 0], [1, 1]])
        y = x.gather(dim, index)
        print(y)

        x = torch.randn(3, 4)
        dim = 1
        index = torch.tensor([[0, 0], [1, 1]])
        y = x.gather(dim, index)
        print(y)

        assert y.shape == (2, 2)

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.full((2, 2), inf),
            'nan': torch.full((2, 2), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            idx = torch.Tensor().npu()
            y = x.gather(dim=0, index=idx)
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            idx = torch.tensor([[0, 0], [1, 1]]).npu()
            y = x.gather(dim=1, index=idx)
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            idx = torch.tensor([[0, 0], [1, 1]]).npu()
            y = x.gather(dim=1, index=idx)
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            idx = torch.Tensor([]).to(torch.int32).npu()
            y = x.gather(dim=0, index=idx)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

            x = torch.full((2, 3), inf).npu()
            idx = torch.tensor([[0, 0], [1, 1]]).npu()
            y = x.gather(dim=1, index=idx)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([2, 2]) and torch.isinf(y).all()

            x = torch.full((2, 3), nan).npu()
            idx = torch.tensor([[0, 0], [1, 1]]).npu()
            y = x.gather(dim=1, index=idx)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([2, 2]) and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0859,
            (100, 100):   0.0871,
            (1000, 100):  0.0937,
            (1000, 1000): 0.0951,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.3596,
            (100, 100):   0.4163,
            (1000, 100):  2.5883,
            (1000, 1000): 3.0336,
        }

        pt_max_memory = {
            (10, 10):     16781824.0,
            (100, 100):   16856064.0,
            (1000, 100):  17530368.0,
            (1000, 1000): 24876544.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            idx = torch.randint(0, min(shape), (shape[0] // 2, shape[1] // 2)).npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.gather(dim=1, index=idx)
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

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            idx = torch.randint(0, min(shape), (shape[0] // 2, shape[1] // 2)).npu()

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.gather(dim=1, index=idx)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 8.00:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 8.00), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 12.6: # runtime problem
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.26), but got '
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
        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)
        idx = torch.randint(0, 100, (50, 50)).npu()

        # forward and backward
        y = x.gather(dim=1, index=idx)
        y = y.sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'idx': idx.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x, y_pt, x_grad_pt, idx = pt_data['x'], pt_data['y'], pt_data['x_grad'], pt_data['idx']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.gather(dim=1, index=idx)
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


class TestIndexSelect(unittest.TestCase):
    ops_name = "index_select"
    def test_api_completeness(self):
        x = torch.randn(3, 4)
        dim = 0
        index = torch.tensor([0, 2])
        y = x.index_select(dim, index)
        assert y.shape == (2, 4)

        x = torch.randn(3, 4)
        dim = 1
        index = torch.tensor([2, 3, 0, 0])
        y = x.index_select(dim, index)

        assert y.shape == (3, 4)

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.full((2, 2), inf),
            'nan': torch.full((2, 2), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            idx = torch.Tensor().int().npu()
            y = x.index_select(dim=0, index=idx)
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            idx = torch.tensor([0, 2]).npu()
            y = x.index_select(dim=1, index=idx)
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            idx = torch.tensor([0, 2]).npu()
            y = x.index_select(dim=1, index=idx)
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            idx = torch.Tensor([]).to(torch.int32).npu()
            y = x.gather(dim=0, index=idx)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

            x = torch.full((2, 3), inf).npu()
            idx = torch.tensor([[0, 0], [1, 1]]).npu()
            y = x.gather(dim=1, index=idx)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([2, 2]) and torch.isinf(y).all()

            x = torch.full((2, 3), nan).npu()
            idx = torch.tensor([[0, 0], [1, 1]]).npu()
            y = x.gather(dim=1, index=idx)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([2, 2]) and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0805,
            (100, 100):   0.0804,
            (1000, 100):  0.0896,
            (1000, 1000): 0.0914,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.3568,
            (100, 100):   0.4088,
            (1000, 100):  1.9050,
            (1000, 1000): 2.6958,
        }

        pt_max_memory = {
            (10, 10):     16783872.0,
            (100, 100):   16833536.0,
            (1000, 100):  17291264.0,
            (1000, 1000): 21791744.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            idx = torch.randint(0, min(shape), (shape[0] // 2,)).npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.index_select(dim=0, index=idx)
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

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            idx = torch.randint(0, min(shape), (shape[0] // 2, )).npu()

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.index_select(dim=0, index=idx)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 4.45:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 4.45), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 12.8: # runtime problem
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.28), but got '
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
        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        idx = torch.randint(0, 100, (50,)).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.index_select(dim=1, index=idx)
        y = y.sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'idx': idx.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x, y_pt, x_grad_pt, idx = pt_data['x'], pt_data['y'], pt_data['x_grad'], pt_data['idx']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.index_select(dim=1, index=idx)
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


class TestNonzero(unittest.TestCase):
    ops_name = "nonzero"
    def test_api_completeness(self):
        x = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                          [0.0, 0.4, 0.0, 0.0],
                          [0.0, 0.0, 1.2, 0.0],
                          [0.0, 0.0, 0.0, -0.4]])
        y = x.nonzero()
        print(y)
        assert torch.equal(y, torch.tensor([[0, 0],
                                            [1, 1],
                                            [2, 2],
                                            [3, 3]]))

        x = torch.tensor([[[0.6, 0.0, 0.0, 0.0],
                           [0.0, 0.4, 0.0, 0.0],
                           [0.0, 0.0, 1.2, 0.0],
                           [0.0, 0.0, 0.0, -0.4]],

                          [[0.6, 0.0, 0.0, 0.0],
                           [0.0, 0.4, 0.0, 0.0],
                           [0.0, 0.0, 1.2, 0.0],
                           [0.0, 0.0, 0.0, -0.4]]])
        y = x.nonzero(as_tuple=True)
        print(y)

    def test_api_outlier(self, backend):

        pt_output = {
            'inf': torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).npu(),
            'nan': torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).npu()
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.nonzero()
            print('torch empty: ', y)

            x = torch.full((2, 2), inf).npu()
            y = x.nonzero()
            print('torch inf: ', y)

            x = torch.full((2, 2), nan).npu()
            y = x.nonzero()
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.nonzero()
            print('ms empty: ', y)
            assert y.shape == torch.Size([0, 1])

            x = torch.full((2, 2), inf).npu()
            y = x.nonzero()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert torch.equal(y, pt_output['inf'])

            x = torch.full((2, 2), nan).npu()
            y = x.nonzero()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert torch.equal(y, pt_output['nan'])

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.1740,
            (100, 100):   0.2487,
            (1000, 100):  0.5333,
            (1000, 1000): 3.4264,
        }

        pt_max_memory = {
            (10, 10):     8704.0,
            (100, 100):   465920.0,
            (1000, 100):  4438016.0,
            (1000, 1000): 45559808.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            zeros_mask = torch.randint_like(x, 0, 2, dtype=x.dtype)
            x = x * zeros_mask
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.nonzero()

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

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
            zeros_mask = torch.randint_like(x, 0, 2, dtype=x.dtype)
            x = x * zeros_mask

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()

                y = x.nonzero()

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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 1.3: # runtime problem
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.13), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 0.92:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.92), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        zeros_mask = torch.randint_like(x, 0, 2, dtype=x.dtype)
        x = x * zeros_mask

        # forward and backward
        y = x.nonzero()
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

        y_ms = x_pt.nonzero()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"


class TestPermute(unittest.TestCase):
    ops_name = "permute"
    def test_api_completeness(self):
        x = torch.randn(1, 2, 3, 4)
        y = x.permute(2, 0, 1, 3)
        assert y.shape == torch.Size([3, 1, 2, 4])

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.full((3, 2), inf),
            'nan': torch.full((3, 2), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.permute(0)
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            y = x.permute(1, 0)
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            y = x.permute(1, 0)
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.permute(0)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

            x = torch.full((2, 3), inf).npu()
            y = x.permute(1, 0)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([3, 2]) and torch.isinf(y).all()

            x = torch.full((2, 3), nan).npu()
            y = x.permute(1, 0)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([3, 2]) and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0737,
            (100, 100):   0.0712,
            (1000, 100):  0.0779,
            (1000, 1000): 0.0810,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2408,
            (100, 100):   0.2981,
            (1000, 100):  2.0185,
            (1000, 1000): 3.0625,
        }

        pt_max_memory = {
            (10, 10):     4096.0,
            (100, 100):   44032.0,
            (1000, 100):  414720.0,
            (1000, 1000): 4015104.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.permute(1, 0)
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
                y = x.permute(1, 0)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 5.31:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 5.31), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.85:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.85), but got '
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
        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        idx = torch.randint(0, 100, (50,)).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.permute(1, 0)
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

        x, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.permute(1, 0)
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


class TestReshape(unittest.TestCase):
    ops_name = "reshape"
    def test_api_completeness(self):
        x = torch.randn(1, 2, 3, 4)
        y = x.reshape(4, 6)
        assert y.shape == torch.Size((4, 6))

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.full((6, 1), inf),
            'nan': torch.full((6, 1), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.reshape(3, 0)
            print('torch empty: ', y)

            x = torch.full((2, 3), inf).npu()
            y = x.reshape(6, 1)
            print('torch inf: ', y)

            x = torch.full((2, 3), nan).npu()
            y = x.reshape(6, 1)
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.reshape(3, 0)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([3, 0])

            x = torch.full((2, 3), inf).npu()
            y = x.reshape(6, 1)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([6, 1]) and torch.isinf(y).all()

            x = torch.full((2, 3), nan).npu()
            y = x.reshape(6, 1)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([6, 1]) and torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0647,
            (100, 100):   0.0630,
            (1000, 100):  0.0775,
            (1000, 1000): 0.0712,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2402,
            (100, 100):   0.3060,
            (1000, 100):  1.9690,
            (1000, 1000): 3.0713,
        }

        pt_max_memory = {
            (10, 10):     2560.0,
            (100, 100):   41984.0,
            (1000, 100):  401408.0,
            (1000, 1000): 4001792.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.reshape(50, -1)
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
                y = x.reshape(50, -1)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 6.06:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 6.06), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.95:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.95), but got '
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
        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.reshape(50, -1)
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

        x, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.reshape(50, -1)
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


class TestScatter(unittest.TestCase):
    ops_name = "scatter"
    def test_api_completeness(self):
        dim = 0
        index = torch.tensor([[0, 1, 2, 0]])
        src = torch.arange(1, 11).reshape((2, 5))
        x = torch.zeros(3, 5, dtype=src.dtype)
        y = x.scatter(dim, index, src)
        print(y)

        dim = 1
        index = torch.tensor([[0, 2, 4], [0, 2, 4], [0, 2, 4]])
        src = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.bfloat16)
        x = torch.zeros(5, 5, dtype=src.dtype)
        y = x.scatter(dim, index, src)
        print(y)

        expected_result = torch.tensor([[1., 0., 2., 0., 3.],
                                        [4., 0., 5., 0., 6.],
                                        [7., 0., 8., 0., 9.],
                                        [0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0.]], dtype=torch.bfloat16)
        assert torch.equal(y, expected_result)

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.tensor([[9., 8.], [4., inf], [inf, 1.]]).npu(),
            'nan': torch.tensor([[9., 8.], [4., nan], [nan, 1.]]).npu()
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            idx = torch.Tensor().long().npu()
            src = torch.Tensor().npu()
            y = x.scatter(0, idx, src)
            print('torch empty: ', y)

            x = torch.full((3, 2), inf).npu()
            idx = torch.tensor([[0, 1], [2, 0], [1, 2]]).long().npu()
            src = torch.tensor([[9.0, 8.0], [5.0, 4.0], [1.0, 0.0]]).npu()
            y = x.scatter(1, idx, src)
            print('torch inf: ', y)

            x = torch.full((3, 2), nan).npu()
            idx = torch.tensor([[0, 1], [2, 0], [1, 2]]).long().npu()
            src = torch.tensor([[9.0, 8.0], [5.0, 4.0], [1.0, 0.0]]).npu()
            y = x.scatter(1, idx, src)
            print('torch nan: ', y)

        elif backend == 'mindspore':

            # not support for null tensor
            # x = torch.Tensor([]).npu()
            # idx = torch.Tensor([]).long().npu()
            # src = torch.Tensor([]).npu()
            # y = x.scatter(0, idx, src)
            # print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            # assert y.shape == torch.Size([0])

            x = torch.full((3, 2), inf).npu()
            idx = torch.tensor([[0, 1], [2, 0], [1, 2]]).long().npu()
            src = torch.tensor([[9.0, 8.0], [5.0, 4.0], [1.0, 0.0]]).npu()
            y = x.scatter(1, idx, src)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])

            inf_mask = torch.isinf(y)

            assert torch.equal(y[~inf_mask], pt_output['inf'][~inf_mask]) and \
                   len(inf_mask) == len(torch.isinf(pt_output['inf']))

            x = torch.full((3, 2), nan).npu()
            idx = torch.tensor([[0, 1], [2, 0], [1, 2]]).long().npu()
            src = torch.tensor([[9.0, 8.0], [5.0, 4.0], [1.0, 0.0]]).npu()
            y = x.scatter(1, idx, src)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            nan_mask = torch.isnan(y)
            assert torch.equal(y[~nan_mask], pt_output['nan'][~nan_mask]) and \
                   len(nan_mask) == len(torch.isnan(pt_output['nan']))


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.1700,
            (100, 100):   0.1728,
            (1000, 100):  0.1864,
            (1000, 1000): 0.0991,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.4154,
            (100, 100):   0.4605,
            (1000, 100):  2.0829,
            (1000, 1000): 3.0259,
        }

        pt_max_memory = {
            (10, 10):     16782848.0,
            (100, 100):   16881664.0,
            (1000, 100):  17780736.0,
            (1000, 1000): 27473920.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            src = torch.randn((shape[0]//2, shape[1]//2), dtype=x.dtype).npu()
            idx = torch.randint_like(src, 0, shape[1]).long().npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.scatter(1, idx, src)
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

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            src = torch.randn((shape[0]//2, shape[1]//2), dtype=x.dtype).npu()
            idx = torch.randint_like(src, 0, shape[1]).long().npu()
            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.scatter(1, idx, src)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 4.58:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 4.58), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 10.7: # runtime problem
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.07), but got '
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
        shape = (100, 100)
        x = torch.randn(shape, dtype=dtype, requires_grad=True).npu()
        src = torch.randn((shape[0]//2, shape[1]//2), dtype=x.dtype).npu()
        idx = torch.randint_like(src, 0, shape[1]).long().npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.scatter(1, idx, src)
        y = y.sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'src': src.contiguous(),
            'idx': idx.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x, src, idx, y_pt, x_grad_pt = pt_data['x'], pt_data['src'], pt_data['idx'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.scatter(1, idx, src)
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


class TestScatterAdd(unittest.TestCase):
    ops_name = "scatter_add"
    def test_api_completeness(self):
        dim = 0
        index = torch.tensor([[0, 1, 2, 0]])
        src = torch.arange(1, 11).reshape((2, 5))
        x = torch.ones(3, 5, dtype=src.dtype)
        y = x.scatter_add(dim, index, src)
        print(y)

        dim = 1
        index = torch.tensor([[0, 2, 4], [0, 2, 4], [0, 2, 4]])
        src = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.bfloat16)
        x = torch.ones(5, 5, dtype=src.dtype)
        y = x.scatter_add(dim, index, src)
        print(y)

        expected_result = torch.tensor([[ 2,  1,  3,  1,  4],
                                        [ 5,  1,  6,  1,  7],
                                        [ 8,  1,  9,  1, 10],
                                        [ 1,  1,  1,  1,  1],
                                        [ 1,  1,  1,  1,  1]], dtype=torch.bfloat16)
        assert torch.equal(y, expected_result)

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.full((3, 2), inf),
            'nan': torch.full((3, 2), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            idx = torch.Tensor().long().npu()
            src = torch.Tensor().npu()
            y = x.scatter_add(0, idx, src)
            print('torch empty: ', y)

            x = torch.full((3, 2), inf).npu()
            idx = torch.tensor([[0, 1], [2, 0], [1, 2]]).long().npu()
            src = torch.tensor([[9.0, 8.0], [5.0, 4.0], [1.0, 0.0]]).npu()
            y = x.scatter_add(1, idx, src)
            print('torch inf: ', y)

            x = torch.full((3, 2), nan).npu()
            idx = torch.tensor([[0, 1], [2, 0], [1, 2]]).long().npu()
            src = torch.tensor([[9.0, 8.0], [5.0, 4.0], [1.0, 0.0]]).npu()
            y = x.scatter_add(1, idx, src)
            print('torch nan: ', y)

        elif backend == 'mindspore':

            # not support for null tensor
            # x = torch.Tensor([]).npu()
            # idx = torch.Tensor([]).long().npu()
            # src = torch.Tensor([]).npu()
            # y = x.scatter_add(0, idx, src)
            # print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            # assert y.shape == torch.Size([0])

            x = torch.full((3, 2), inf).npu()
            idx = torch.tensor([[0, 1], [2, 0], [1, 2]]).long().npu()
            src = torch.tensor([[9.0, 8.0], [5.0, 4.0], [1.0, 0.0]]).npu()
            y = x.scatter_add(1, idx, src)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([3, 2]) and torch.isinf(y).all()

            x = torch.full((3, 2), nan).npu()
            idx = torch.tensor([[0, 1], [2, 0], [1, 2]]).long().npu()
            src = torch.tensor([[9.0, 8.0], [5.0, 4.0], [1.0, 0.0]]).npu()
            y = x.scatter_add(1, idx, src)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([3, 2]) and torch.isnan(y).all()


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.1883,
            (100, 100):   0.1886,
            (1000, 100):  0.2329,
            (1000, 1000): 0.1847,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2170,
            (100, 100):   0.2677,
            (1000, 100):  2.6469,
            (1000, 1000): 3.4332,
        }

        pt_max_memory = {
            (10, 10):     16781312.0,
            (100, 100):   16855552.0,
            (1000, 100):  17529856.0,
            (1000, 1000): 24280576.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            src = torch.randn((shape[0]//2, shape[1]//2), dtype=x.dtype).npu()
            idx = torch.randint_like(src, 0, shape[1]).long().npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.scatter_add(1, idx, src)
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

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            src = torch.randn((shape[0]//2, shape[1]//2), dtype=x.dtype).npu()
            idx = torch.randint_like(src, 0, shape[1]).long().npu()
            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.scatter_add(1, idx, src)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 3.10:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 3.10), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 2.03:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 2.03), but got '
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
        shape = (100, 100)
        x = torch.randn(shape, dtype=dtype, requires_grad=True).npu()
        src = torch.randn((shape[0]//2, shape[1]//2), dtype=x.dtype).npu()
        idx = torch.randint_like(src, 0, shape[0]).long().npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.scatter_add(1, idx, src)
        y = y.sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'src': src.contiguous(),
            'idx': idx.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):
        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x, src, idx, y_pt, x_grad_pt = pt_data['x'], pt_data['src'], pt_data['idx'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.scatter_add(1, idx, src)
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


class TestSelect(unittest.TestCase):
    ops_name = "select"
    def test_api_completeness(self):
        dim = 0
        index = 1
        x = torch.arange(1, 11).reshape((2, 5))
        y = x.select(dim, index)
        print(y)

        dim = 1
        index = 3
        y = x.select(dim, index)
        print(y)

        expected_result = torch.tensor([4, 9], dtype=x.dtype)
        assert torch.equal(y, expected_result)

    def test_api_outlier(self, backend):

        pt_output = {
            'inf': torch.full((2,), inf),
            'nan': torch.full((2,), nan)
        }

        if backend == 'torch':

            x = torch.full((3, 2), inf).npu()
            y = x.select(0, 2)
            print('torch inf: ', y)

            x = torch.full((3, 2), nan).npu()
            y = x.select(0, 2)
            print('torch nan: ', y)

        elif backend == 'mindspore':

            x = torch.full((3, 2), inf).npu()
            y = x.select(0, 2)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([2]) and torch.isinf(y).all()

            x = torch.full((3, 2), nan).npu()
            y = x.select(0, 2)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([2]) and torch.isnan(y).all()


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0648,
            (100, 100):   0.0647,
            (1000, 100):  0.0801,
            (1000, 1000): 0.1003,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2294,
            (100, 100):   0.2870,
            (1000, 100):  2.1714,
            (1000, 1000): 2.9235,
        }

        pt_max_memory = {
            (10, 10):     3072.0,
            (100, 100):   43008.0,
            (1000, 100):  403968.0,
            (1000, 1000): 4004352.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            idx = shape[1] // 2
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.select(1, idx)
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
            idx = shape[1] // 2
            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.select(1, idx)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 5.56:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 5.56), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 2.57:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 2.57), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 1.80:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 1.80), but got '
                    f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []
        def x_hook(grad):
            x_grad.append(grad)
        shape = (100, 100)
        x = torch.randn(shape, dtype=dtype, requires_grad=True).npu()
        idx = shape[1] // 2
        x.register_hook(x_hook)

        # forward and backward
        y = x.select(1, idx)
        y = y.sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'idx': idx,
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x, idx, y_pt, x_grad_pt = pt_data['x'], pt_data['idx'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.select(1, idx)
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


class TestSqueeze(unittest.TestCase):
    ops_name = "squeeze"
    def test_api_completeness(self):
        x = torch.arange(9).reshape((3, 1, 1, 3))
        y = x.squeeze()
        assert y.shape == torch.Size([3, 3])

        y = x.squeeze(1)
        assert y.shape == torch.Size([3, 1, 3])


    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.Tensor([]).reshape(2, 0),
            'inf': torch.full((2,), inf),
            'nan': torch.full((2,), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().reshape(2, 1, 0).npu()
            y = x.squeeze()
            print('torch empty: ', y)

            x = torch.full((3, 1, 2), inf).npu()
            y = x.squeeze()
            print('torch inf: ', y)

            x = torch.full((3, 1, 2), nan).npu()
            y = x.squeeze()
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).reshape(2, 1, 0).npu()
            y = x.squeeze()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([2, 0])

            x = torch.full((3, 1, 2), inf).npu()
            y = x.squeeze()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([3, 2]) and torch.isinf(y).all()

            x = torch.full((3, 1, 2), nan).npu()
            y = x.squeeze()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([3, 2]) and torch.isnan(y).all()


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0596,
            (100, 100):   0.0599,
            (1000, 100):  0.0754,
            (1000, 1000): 0.0798,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2420,
            (100, 100):   0.2946,
            (1000, 100):  2.0207,
            (1000, 1000): 2.9471,
        }

        pt_max_memory = {
            (10, 10):     2560.0,
            (100, 100):   41984.0,
            (1000, 100):  401408.0,
            (1000, 1000): 4001792.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu().unsqueeze(1).unsqueeze(-1)
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.squeeze()
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

            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).unsqueeze(1).unsqueeze(-1)
            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.squeeze()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 6.20:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 6.20), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.87:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.87), but got '
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
        shape = (100, 100)
        x = torch.randn(shape, dtype=dtype, requires_grad=True).npu().unsqueeze(1).unsqueeze(-1)
        x.register_hook(x_hook)

        # forward and backward
        y = x.squeeze()
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

        x, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.squeeze()
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


class TestT(unittest.TestCase):
    ops_name = "t"
    def test_api_completeness(self):
        x = torch.arange(6).reshape((2, 3))
        y = x.t()
        assert y.shape == torch.Size([3, 2])

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.Tensor([]).reshape(2, 0),
            'inf': torch.full((2,), inf),
            'nan': torch.full((2,), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().reshape(2, 0).npu()
            y = x.t()
            print('torch empty: ', y)

            x = torch.full((3, 2), inf).npu()
            y = x.t()
            print('torch inf: ', y)

            x = torch.full((3, 2), nan).npu()
            y = x.t()
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).reshape(2, 0).npu()
            y = x.t()
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0, 2])

            x = torch.full((3, 2), inf).npu()
            y = x.t()
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([2, 3]) and torch.isinf(y).all()

            x = torch.full((3, 2), nan).npu()
            y = x.t()
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([2, 3]) and torch.isnan(y).all()


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0667,
            (100, 100):   0.0630,
            (1000, 100):  0.0739,
            (1000, 1000): 0.0806,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2470,
            (100, 100):   0.3065,
            (1000, 100):  1.9654,
            (1000, 1000): 3.0674,
        }

        pt_max_memory = {
            (10, 10):     4096.0,
            (100, 100):   44032.0,
            (1000, 100):  414720.0,
            (1000, 1000): 4015104.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.t()
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
                y = x.t()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 5.37:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 5.37), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.85:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.85), but got '
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
        shape = (100, 100)
        x = torch.randn(shape, dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.t()
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

        x, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.t()
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


class TestTranspose(unittest.TestCase):
    ops_name = "transpose"
    def test_api_completeness(self):
        x = torch.arange(6).reshape((2, 1, 3))
        y = x.transpose(2, 1)
        assert y.shape == torch.Size([2, 3, 1])

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.Tensor([]).reshape(2, 0),
            'inf': torch.full((2,), inf),
            'nan': torch.full((2,), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().reshape(2, 0).npu()
            y = x.transpose(0, 1)
            print('torch empty: ', y)

            x = torch.full((3, 2), inf).npu()
            y = x.transpose(0, 1)
            print('torch inf: ', y)

            x = torch.full((3, 2), nan).npu()
            y = x.transpose(0, 1)
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).reshape(2, 0).npu()
            y = x.transpose(0, 1)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0, 2])

            x = torch.full((3, 2), inf).npu()
            y = x.transpose(0, 1)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([2, 3]) and torch.isinf(y).all()

            x = torch.full((3, 2), nan).npu()
            y = x.transpose(0, 1)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([2, 3]) and torch.isnan(y).all()


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0686,
            (100, 100):   0.0678,
            (1000, 100):  0.0707,
            (1000, 1000): 0.0750,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2530,
            (100, 100):   0.2958,
            (1000, 100):  1.6668,
            (1000, 1000): 2.6379,
        }

        pt_max_memory = {
            (10, 10):     4096.0,
            (100, 100):   44032.0,
            (1000, 100):  414720.0,
            (1000, 1000): 4015104.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.transpose(0, 1)
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
                y = x.transpose(0, 1)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 5.67:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 5.67), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.90:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.90), but got '
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
        shape = (100, 100)
        x = torch.randn(shape, dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.transpose(0, 1)
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

        x, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.transpose(0, 1)
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


class TestView(unittest.TestCase):
    ops_name = "view"
    def test_api_completeness(self):
        x = torch.randn((1, 2, 3, 4, 5))
        y = x.view(6, 20)
        assert y.shape == torch.Size([6, 20])

        y = x.view(3, -1)
        assert y.shape == torch.Size([3, 40])

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.Tensor([]).reshape(2, 0),
            'inf': torch.full((2,), inf),
            'nan': torch.full((2,), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().reshape(2, 0).npu()
            y = x.view(-1)
            print('torch empty: ', y)

            x = torch.full((3, 2), inf).npu()
            y = x.view(-1)
            print('torch inf: ', y)

            x = torch.full((3, 2), nan).npu()
            y = x.view(-1)
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).reshape(2, 0).npu()
            y = x.view(-1)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

            x = torch.full((3, 2), inf).npu()
            y = x.view(-1)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([6]) and torch.isinf(y).all()

            x = torch.full((3, 2), nan).npu()
            y = x.view(-1)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([6]) and torch.isnan(y).all()


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0607,
            (100, 100):   0.0618,
            (1000, 100):  0.0757,
            (1000, 1000): 0.0814,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2377,
            (100, 100):   0.3124,
            (1000, 100):  1.9320,
            (1000, 1000): 2.7989,
        }

        pt_max_memory = {
            (10, 10):     2560.0,
            (100, 100):   41984.0,
            (1000, 100):  401408.0,
            (1000, 1000): 4001792.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.view(-1)
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
                y = x.view(-1)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 5.89:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 5.89), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.84:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.84), but got '
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
        shape = (100, 100)
        x = torch.randn(shape, dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.view(-1)
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

        x, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.view(-1)
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


class TestView(unittest.TestCase):
    ops_name = "view"
    def test_api_completeness(self):
        x = torch.randn((1, 2, 3, 4, 5))
        y = x.view(6, 20)
        assert y.shape == torch.Size([6, 20])

        y = x.view(3, -1)
        assert y.shape == torch.Size([3, 40])

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.Tensor([]).reshape(2, 0),
            'inf': torch.full((2,), inf),
            'nan': torch.full((2,), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().reshape(2, 0).npu()
            y = x.view(-1)
            print('torch empty: ', y)

            x = torch.full((3, 2), inf).npu()
            y = x.view(-1)
            print('torch inf: ', y)

            x = torch.full((3, 2), nan).npu()
            y = x.view(-1)
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).reshape(2, 0).npu()
            y = x.view(-1)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([0])

            x = torch.full((3, 2), inf).npu()
            y = x.view(-1)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([6]) and torch.isinf(y).all()

            x = torch.full((3, 2), nan).npu()
            y = x.view(-1)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([6]) and torch.isnan(y).all()


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0607,
            (100, 100):   0.0618,
            (1000, 100):  0.0757,
            (1000, 1000): 0.0814,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2377,
            (100, 100):   0.3124,
            (1000, 100):  1.9320,
            (1000, 1000): 2.7989,
        }

        pt_max_memory = {
            (10, 10):     2560.0,
            (100, 100):   41984.0,
            (1000, 100):  401408.0,
            (1000, 1000): 4001792.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.view(-1)
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
                y = x.view(-1)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 5.89:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 5.89), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.84:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.84), but got '
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
        shape = (100, 100)
        x = torch.randn(shape, dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.view(-1)
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

        x, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.view(-1)
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


class TestViewAs(unittest.TestCase):
    ops_name = "view_as"
    def test_api_completeness(self):
        x = torch.randn((1, 2, 3, 4, 5))
        y = x.view(6, 20)
        y = x.view_as(y)
        assert y.shape == torch.Size([6, 20])

        y = x.view(3, -1)
        y = x.view_as(y)
        assert y.shape == torch.Size([3, 40])

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.Tensor([]).reshape(2, 0),
            'inf': torch.full((2,), inf),
            'nan': torch.full((2,), nan)
        }

        if backend == 'torch':
            x = torch.Tensor().reshape(2, 0).npu()
            src = torch.Tensor().reshape(3, 0).npu()
            y = x.view_as(src)
            print('torch empty: ', y)

            x = torch.full((3, 2), inf).npu()
            src = torch.empty((3, 1, 2)).npu()
            y = x.view_as(src)
            print('torch inf: ', y)

            x = torch.full((3, 2), nan).npu()
            y = x.view_as(src)
            print('torch nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).reshape(2, 0).npu()
            src = torch.Tensor([]).reshape(3, 0).npu()
            y = x.view_as(src)
            print('ms empty: ', y, ' torch empty: ', pt_output['empty'])
            assert y.shape == torch.Size([3, 0])

            x = torch.full((3, 2), inf).npu()
            src = torch.empty((3, 1, 2)).npu()
            y = x.view_as(src)
            print('ms inf: ', y, ' torch inf: ', pt_output['inf'])
            assert y.shape == torch.Size([3, 1, 2]) and torch.isinf(y).all()

            x = torch.full((3, 2), nan).npu()
            y = x.view_as(src)
            print('ms nan: ', y, ' torch nan: ', pt_output['nan'])
            assert y.shape == torch.Size([3, 1, 2]) and torch.isnan(y).all()


    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10):     0.0635,
            (100, 100):   0.0635,
            (1000, 100):  0.0719,
            (1000, 1000): 0.0761,
        }

        pt_backward_time_diff_shape = {
            (10, 10):     0.2457,
            (100, 100):   0.3066,
            (1000, 100):  2.1911,
            (1000, 1000): 2.7470,
        }

        pt_max_memory = {
            (10, 10):     3072.0,
            (100, 100):   62464.0,
            (1000, 100):  601600.0,
            (1000, 1000): 6002176.0,
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            src = torch.empty_like(x).reshape(50, -1)
            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.view_as(src)
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
            src = torch.empty_like(x).reshape(50, -1)
            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.view_as(src)
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 5.19:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 5.19), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.80:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.80), but got '
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
        shape = (100, 100)
        x = torch.randn(shape, dtype=dtype, requires_grad=True).npu()
        src = torch.empty_like(x).reshape(50, -1)
        x.register_hook(x_hook)

        # forward and backward
        y = x.view_as(src)
        y = y.sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'src': src.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict,
                   os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_{self.ops_name}_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x, src, y_pt, x_grad_pt = pt_data['x'], pt_data['src'], pt_data['y'], pt_data['x_grad']

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)
        x.register_hook(weight_hook)
        # forward
        y_ms = x.view_as(src)
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
    TestList = [TestChunk, TestGather, TestIndexSelect, TestNonzero, TestPermute, TestReshape, TestScatter,
                TestScatterAdd, TestSelect, TestSqueeze, TestT, TestTranspose, TestView, TestViewAs]

    test_precision_dtype = [torch.bfloat16, torch.float16]
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
