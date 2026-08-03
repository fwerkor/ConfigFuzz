import os
import time
import random
import argparse
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import numpy as np

pt_input_output_path = "/home/workspace/mindspore_dataset/msadapter/test_input/ut/tensor/test_reduction"

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


class TestAll(TestCase):

    def test_api_completeness(self):
        x = torch.rand(1, 2).bool()
        y = x.all()
        print(y)

        x = torch.rand(1, 2).bool()
        y = x.all()
        print(y)

        x = torch.arange(0, 3)
        y = x.all()
        print(y)

        x = torch.rand(4, 2).bool()
        dim = 0
        y = x.all(dim=dim)
        print(y)

        x = torch.ones(4, 2).bool()
        dim = 1
        y = x.all(dim=dim, keepdim=True)
        print(y)

        x = torch.ones(4, 2)
        assert x.all()

        x[0][1] = 0
        assert not x.all()

    def test_api_outlier(self, backend):
        # no torch.empty test because torch.empty generate a random tensor,
        # means nothing for 'all' ops

        pt_output = {
            'empty': torch.tensor(True),
            'inf': torch.tensor(True),
            'nan': torch.tensor(True)
        }

        if backend == 'torch':

            x = torch.Tensor().npu()
            y = x.all()
            print('torch all empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.all()
            print('torch all inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.all()
            print('torch all nan: ', y)

        elif backend == 'mindspore':

            x = torch.Tensor([]).npu()
            y = x.all()
            print('ms all empty: ', y, ' torch all empty: ', pt_output['empty'])
            assert torch.equal(y, pt_output['empty'])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.all()
            print('ms all inf: ', y, ' torch all inf: ', pt_output['inf'])
            assert torch.equal(y, pt_output['inf'])

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.all()
            print('ms all nan: ', y, ' torch all nan: ', pt_output['nan'])
            assert torch.equal(y, pt_output['nan'])

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 0.3208,
            (100, 100): 0.3172,
            (1000, 100): 0.3334,
            (1000, 1000): 0.3315
        }

        pt_max_memory = {
            (10, 10): 3584.0,
            (100, 100): 33280.0,
            (1000, 100): 314368.0,
            (1000, 1000): 3014656.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            y = x.all().npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.all().npu()
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
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.all()
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

            if forward_cost_time > pt_forward_time_diff_shape[shape] * 1.41:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 1.41), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )
            if max_memory > pt_max_memory[shape] * 5.31:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 5.31), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.all()
        print(f"output is {y}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'y': y
        }

        torch.save(save_dict, os.path.join(pt_input_output_path, f"input_output_all_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data

        pt_data_path = os.path.join(pt_input_output_path, f"input_output_all_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt = pt_data['x'], pt_data['y']

        # forward
        y_ms = x_pt.all()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"


class TestAny(TestCase):

    def test_api_completeness(self):
        x = torch.rand(1, 2).bool()
        y = x.any()
        print(y)

        x = torch.arange(0, 3)
        y = x.any()
        print(y)

        x = torch.randn(4, 2) < 0
        dim = 0
        y = x.any(dim=dim)
        print(y)

        x = torch.randn(4, 2) < 0
        dim = 1
        y = x.any(dim=dim, keepdim=True)
        print(y)

        x = torch.zeros(4, 2)
        assert not x.any()

        x[0][1] = 1
        assert x.any()

    def test_api_outlier(self, backend):
        # no torch.empty test because torch.empty generate a random tensor,
        # means nothing for any ops

        pt_output = {
            'empty': torch.tensor(False),
            'inf': torch.tensor(True),
            'nan': torch.tensor(True)
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.any()
            print('torch any empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.any()
            print('torch any inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.any()
            print('torch any nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.any()
            print('ms any empty: ', y, ' torch any empty: ', pt_output['empty'])
            assert torch.equal(y, pt_output['empty'])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.any()
            print('ms any inf: ', y, ' torch any inf: ', pt_output['inf'])
            assert torch.equal(y, pt_output['inf'])

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.any()
            print('ms any nan: ', y, ' torch any nan: ', pt_output['nan'])
            assert torch.equal(y, pt_output['nan'])

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 0.3441,
            (100, 100): 0.3336,
            (1000, 100): 0.3303,
            (1000, 1000): 0.3413
        }

        pt_max_memory = {
            (10, 10): 3584.0,
            (100, 100): 33280.0,
            (1000, 100): 314368.0,
            (1000, 1000): 3014656.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()
            y = x.any().npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.any().npu()
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
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.any()
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

            if forward_cost_time > pt_forward_time_diff_shape[shape] * 1.97:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 1.97), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )
            if max_memory > pt_max_memory[shape] * 783.43:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 783.43), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.any()
        print(f"output is {y}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'y': y
        }

        torch.save(save_dict, os.path.join(pt_input_output_path, f"input_output_any_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):

        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_any_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt = pt_data['x'], pt_data['y']

        # forward
        y_ms = x_pt.any()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"


class TestArgmax(TestCase):

    def test_api_completeness(self):
        x = torch.randn(4, 4)
        y = x.argmax()
        print(y)

        x = torch.randn(4, 4)
        dim = 0
        y = x.argmax(dim=dim)
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        y = x.argmax(dim=dim, keepdim=True)
        print(y)

        x = torch.zeros(3, 3)
        x[0][1], x[1][2], x[2][0] = 1, 1, 1
        y = x.argmax(dim=dim)
        assert torch.equal(y, torch.tensor([1, 2, 0]))

    def test_api_outlier(self, backend):
        # no torch.empty test because torch.empty generate a random tensor,
        # means nothing for argmax ops

        pt_output = {
            'empty': torch.tensor([]),
            'inf': torch.tensor(0),
            'nan': torch.tensor(0)
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.argmax()
            print('torch argmax empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.argmax()
            print('torch argmax inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.argmax()
            print('torch argmax nan: ', y)

        elif backend == 'mindspore':
            # not support null tensor
            # x = torch.Tensor([]).npu()
            # y = x.argmax()
            # print('ms argmax empty: ', y, ' torch argmax empty: ', pt_output['empty'])
            # assert torch.equal(y, pt_output['empty'])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.argmax()
            print('ms argmax inf: ', y, ' torch argmax inf: ', pt_output['inf'])
            assert torch.equal(y, pt_output['inf'])

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.argmax()
            print('ms argmax nan: ', y, ' torch argmax nan: ', pt_output['nan'])
            assert torch.equal(y, pt_output['nan'])

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 0.3301,
            (100, 100): 0.3346,
            (1000, 100): 0.3336,
            (1000, 1000): 0.4206
        }

        pt_max_memory = {
            (10, 10): 2560.0,
            (100, 100): 22528.0,
            (1000, 100): 202240.0,
            (1000, 1000): 2002432.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.argmax().npu()
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
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)

            for _ in range(repeat_times):
                # forward
                ms.runtime.reset_max_memory_allocated()
                forward_start_time = time.time()

                y = x.argmax()
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

            if forward_cost_time > pt_forward_time_diff_shape[shape] * 3.031:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 3.031), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )
            if max_memory > pt_max_memory[shape] * 2186.40:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 2186.40), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.argmax()
        print(f"output is {y}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x,
            'y': y
        }

        torch.save(save_dict, os.path.join(pt_input_output_path, f"input_output_argmax_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):
        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_argmax_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")
        x_pt, y_pt = pt_data['x'], pt_data['y']

        # forward
        y_ms = x_pt.argmax()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"


class TestMax(TestCase):

    def test_api_completeness(self):
        x = torch.randn(1, 3)
        y = x.max()
        print(y)

        x = torch.randn(4, 4)
        dim = 0
        y = x.max(dim=dim)
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        y = x.max(dim=dim, keepdim=True)
        print(y)

        x = torch.arange(10).reshape(2, 5)
        y = x.max()
        assert y == 9, f"expected result is 9, but got {y}"

    def test_api_outlier(self, backend):
        # no torch.empty test because torch.empty generate a random tensor,
        # means nothing for max ops

        pt_output = {
            'empty': torch.tensor(0.),
            'inf': torch.tensor(float('inf')),
            'nan': torch.tensor(float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.max()
            print('torch max empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.max()
            print('torch max inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.max()
            print('torch max nan: ', y)

        elif backend == 'mindspore':
            # x = torch.Tensor([]).npu()
            # y = x.max()
            # print('ms max empty: ', y, ' torch max empty: ', pt_output['empty'])
            # assert torch.equal(y, pt_output['empty'])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.max()
            print('ms max inf: ', y, ' torch max inf: ', pt_output['inf'])
            assert torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.max()
            print('ms max nan: ', y, ' torch max nan: ', pt_output['nan'])
            assert torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.1854,
            (100, 100): 1.1839,
            (1000, 100): 1.2141,
            (1000, 1000): 1.1996
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.3719,
            (100, 100): 0.4234,
            (1000, 100): 2.5080,
            (1000, 1000): 3.5949
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35557376.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.max().npu()
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
                y = x.max()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 0.47:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.47), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.68:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.68), but got '
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
        y = x.max()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict, os.path.join(pt_input_output_path, f"input_output_max_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):
        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_max_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")
        x_pt, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.max()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        assert np.allclose(weight_grad[0].to(torch.float32), x_grad_pt.to(torch.float32), 0.0, 0.0)


class TestMin(TestCase):

    def test_api_completeness(self):
        x = torch.randn(1, 3)
        y = x.min()
        print(y)

        x = torch.randn(4, 4)
        dim = 0
        y = x.min(dim=dim)
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        y = x.min(dim=dim, keepdim=True)
        print(y)

        x = torch.arange(10).reshape(2, 5)
        assert x.min() == 0

    def test_api_outlier(self, backend):

        pt_output = {
            'empty': torch.tensor(float('nan')),
            'inf': torch.tensor(float('inf')),
            'nan': torch.tensor(float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.min()
            print('torch min empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.min()
            print('torch min inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.min()
            print('torch min nan: ', y)

        elif backend == 'mindspore':
            # x = torch.Tensor([]).npu()
            # y = x.min()
            # print('ms min empty: ', y, ' torch min empty: ', pt_output['empty'])
            # assert torch.isnan(y).all()

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.min()
            print('ms min inf: ', y, ' torch min inf: ', pt_output['inf'])
            assert torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.min()
            print('ms min nan: ', y, ' torch min nan: ', pt_output['nan'])
            assert torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.1822,
            (100, 100): 1.1772,
            (1000, 100): 1.1641,
            (1000, 1000): 1.1612
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.4473,
            (100, 100): 0.5103,
            (1000, 100): 3.0992,
            (1000, 1000): 3.5772
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35557376.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.min().npu()
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
                y = x.min()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 0.51:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.51), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.40:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.40), but got '
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
        y = x.min()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict, os.path.join(pt_input_output_path, f"input_output_min_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):
        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_min_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.min()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        assert np.allclose(weight_grad[0].to(torch.float32), x_grad_pt.to(torch.float32), 0.0, 0.0)


class TestMean(TestCase):

    def test_api_completeness(self):
        x = torch.randn(1, 3)
        y = x.mean()
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float16
        y = x.mean(dim=dim, dtype=dtype)
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float32
        y = x.mean(dim=dim, dtype=dtype)
        print(y)

        x_np = x.numpy()
        y_np = np.mean(x_np, axis=dim)

        y_ms = y.numpy()
        assert np.allclose(y_ms, y_np, 1e-3, 1e-3), f"expect result close to \n{y_np}, \nbut got \n{y_ms}"

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float64
        y = x.mean(dim=dim, dtype=dtype)
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float16
        y = x.mean(dim=dim, dtype=dtype, keepdim=True)
        print(y)

        x = torch.randn(4, 4)
        dim = 0
        dtype = torch.bfloat16
        y = x.mean(dim=dim, dtype=dtype)
        print(y)

    def test_api_outlier(self, backend):
        # no torch.empty test because torch.empty generate a random tensor,
        # means nothing for mean ops

        pt_output = {
            'empty': torch.tensor(float('nan')),
            'inf': torch.tensor(float('inf')),
            'nan': torch.tensor(float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.mean()
            print('torch mean empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.mean()
            print('torch mean inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.mean()
            print('torch mean nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.mean()
            print('ms mean empty: ', y, ' torch mean empty: ', pt_output['empty'])
            assert torch.isnan(y).all()

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.mean()
            print('ms mean inf: ', y, ' torch mean inf: ', pt_output['inf'])
            assert torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.mean()
            print('ms mean nan: ', y, ' torch mean nan: ', pt_output['nan'])
            assert torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.1396,
            (100, 100): 1.1140,
            (1000, 100): 1.1510,
            (1000, 1000): 1.1557
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.2025,
            (100, 100): 0.2531,
            (1000, 100): 2.8243,
            (1000, 1000): 3.3649
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35557376.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.mean().npu()
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
                y = x.mean()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 0.59:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.59), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 2.66:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 2.66), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 0.44:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.44), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.mean()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict, os.path.join(pt_input_output_path, f"input_output_mean_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):
        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_mean_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.mean()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        assert np.allclose(weight_grad[0].to(torch.float32), x_grad_pt.to(torch.float32), 0.0, 0.0)


class TestNorm(TestCase):

    def test_api_completeness(self):
        x = torch.randn(2, 3)
        y = x.norm()
        y = y.numpy()

        y = x.norm(p='fro', dim=None, keepdim=False, dtype=None)
        y = y.numpy()

        y = x.norm(1, 1, True, torch.float32)
        y = y.numpy()

        y = x.to(torch.bfloat16).norm(3, 0, True, torch.bfloat16)
        y = y.numpy()

        y = x.to(torch.float32).norm(2, 1, True, torch.float32)
        y = y.numpy()

    def test_api_outlier(self, backend):
        # no torch.empty test because torch.empty generate a random tensor,
        # means nothing for norm ops

        pt_output = {
            'empty': torch.tensor(0.),
            'inf': torch.tensor(float('inf')),
            'nan': torch.tensor(float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.norm()
            print('torch norm empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.norm()
            print('torch norm inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.norm()
            print('torch norm nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.norm()
            print('ms norm empty: ', y, ' torch norm empty: ', pt_output['empty'])
            assert torch.equal(y, pt_output['empty'])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.norm()
            print('ms norm inf: ', y, ' torch norm inf: ', pt_output['inf'])
            assert torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.norm()
            print('ms norm nan: ', y, ' torch norm nan: ', pt_output['nan'])
            assert torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.1556,
            (100, 100): 1.0778,
            (1000, 100): 1.1130,
            (1000, 1000): 1.1269
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.2311,
            (100, 100): 0.2830,
            (1000, 100): 3.1026,
            (1000, 1000): 3.5163
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35557376.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.norm().npu()
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
                y = x.norm()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 0.70:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.70), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 3.15:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 3.15), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 0.37:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.37), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.norm()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict, os.path.join(pt_input_output_path, f"input_output_norm_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):
        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_norm_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.norm()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        assert np.allclose(weight_grad[0].to(torch.float32), x_grad_pt.to(torch.float32), 1e-3, 1e-3)


class TestProd(TestCase):

    def test_api_completeness(self):
        x = torch.randn(1, 3)
        y = x.prod()
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float16
        y = x.prod(dim=dim, dtype=dtype)
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float32
        y = x.prod(dim=dim, dtype=dtype)
        print(y)

        x_np = x.numpy()
        y_np = np.prod(x_np, axis=dim)

        y_ms = y.numpy()
        assert np.allclose(y_ms, y_np, 1e-3, 1e-3), f"expect result close to \n{y_np}, \nbut got \n{y_ms}"

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float64
        y = x.prod(dim=dim, dtype=dtype)
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float16
        y = x.prod(dim=dim, dtype=dtype, keepdim=True)
        print(y)

        x = torch.randn(4, 4)
        dim = 0
        dtype = torch.bfloat16
        y = x.prod(dim=dim, dtype=dtype)
        print(y)

    def test_api_outlier(self, backend):
        # no torch.empty test because torch.empty generate a random tensor,
        # means nothing for prod ops

        pt_output = {
            'empty': torch.tensor(1.),
            'inf': torch.tensor(float('inf')),
            'nan': torch.tensor(float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.prod()
            print('torch prod empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.prod()
            print('torch prod inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.prod()
            print('torch prod nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.prod()
            print('ms prod empty: ', y, ' torch prod empty: ', pt_output['empty'])
            assert torch.equal(y, pt_output['empty'])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.prod()
            print('ms prod inf: ', y, ' torch prod inf: ', pt_output['inf'])
            assert torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.prod()
            print('ms prod nan: ', y, ' torch prod nan: ', pt_output['nan'])
            assert torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.1572,
            (100, 100): 0.5461,
            (1000, 100): 0.6023,
            (1000, 1000): 0.6119
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.3970,
            (100, 100): 0.4573,
            (1000, 100): 3.0039,
            (1000, 1000): 3.9138
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35557376.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.prod().npu()
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
                y = x.prod()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 10 * 0.85:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.85), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 10 * 2.1:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 2.1), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 2000 * 0.54:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.54), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.prod()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict, os.path.join(pt_input_output_path, f"input_output_prod_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):
        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_prod_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.prod()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        assert np.allclose(weight_grad[0].to(torch.float32), x_grad_pt.to(torch.float32), 0.0, 0.0)


class TestStd(TestCase):

    def test_api_completeness(self):
        x = torch.randn(2, 3)
        y = x.std()
        print(y)

        x_np = x.numpy()
        y_np = np.std(x_np)

        y_ms = y.numpy()
        assert np.allclose(y_ms, y_np, 1e-3, 1e-3), f"expect result close to \n{y_np}, \nbut got \n{y_ms}"

        y = x.std(dim=1, correction=1, keepdim=False)
        print(y)

        y = x.std(dim=0, keepdim=False)
        print(y)

        y = x.std(dim=0)
        print(y)

        x = torch.randn(4, 4).to(torch.float64)
        y = x.std(dim=1)
        print(y)

        x = torch.randn(4, 4).to(torch.float16)
        y = x.std(dim=1, keepdim=True)
        print(y)

        x = torch.randn(4, 4).to(torch.bfloat16)
        y = x.std(dim=1, keepdim=True)
        print(y)

    def test_api_outlier(self, backend):
        # no torch.empty test because torch.empty generate a random tensor,
        # means nothing for prod ops

        pt_output = {
            'empty': torch.tensor(float('nan')),
            'inf': torch.tensor(float('inf')),
            'nan': torch.tensor(float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.std()
            print('torch std empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.std()
            print('torch std inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.std()
            print('torch std nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.std()
            print('ms std empty: ', y, ' torch std empty: ', pt_output['inf'])
            assert torch.isnan(y).all()

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.std()
            print('ms std inf: ', y, ' torch std inf: ', pt_output['inf'])
            assert torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.std()
            print('ms std nan: ', y, ' torch std nan: ', pt_output['nan'])
            assert torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 1.1933,
            (100, 100): 1.1436,
            (1000, 100): 1.2324,
            (1000, 1000): 1.2013
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.2420,
            (100, 100): 0.3012,
            (1000, 100): 2.7789,
            (1000, 1000): 3.5474
        }

        pt_max_memory = {
            (10, 10): 33557504.0,
            (100, 100): 33577472.0,
            (1000, 100): 33757184.0,
            (1000, 1000): 35557376.0
        }

        if backend == 'torch':
            x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).npu()

            for _ in range(repeat_times):
                torch.npu.reset_max_memory_allocated()
                forward_start_time = time.time()
                y = x.std().npu()
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
                y = x.std()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 10 * 0.85:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.85), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 10 * 2.1:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 2.1), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 2000 * 0.54:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.54), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.std()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict, os.path.join(pt_input_output_path, f"input_output_std_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):
        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_std_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.std()

        # backward
        y_ms.backward()

        # check forward precision
        print(f"ms output is {y_ms}, pt output is {y_pt}")
        assert torch.equal(y_ms, y_pt), f"test {self.__class__.__name__} failed"

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{x_grad_pt}")
        assert np.allclose(weight_grad[0].to(torch.float32), x_grad_pt.to(torch.float32), 0.0, 0.0)


class TestSum(TestCase):

    def test_api_completeness(self):
        x = torch.randn(1, 3)
        y = x.sum()
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float16
        y = x.sum(dim=dim, dtype=dtype)
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float32
        y = x.sum(dim=dim, dtype=dtype)
        print(y)

        x_np = x.numpy()
        y_np = np.sum(x_np, axis=dim)

        y_ms = y.numpy()
        assert np.allclose(y_ms, y_np, 1e-3, 1e-3), f"expect result close to \n{y_np}, \nbut got \n{y_ms}"

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float64
        y = x.sum(dim=dim, dtype=dtype)
        print(y)

        x = torch.randn(4, 4)
        dim = 1
        dtype = torch.float16
        y = x.sum(dim=dim, dtype=dtype, keepdim=True)
        print(y)

        x = torch.randn(4, 4)
        dim = 0
        dtype = torch.bfloat16
        y = x.sum(dim=dim, dtype=dtype)
        print(y)

    def test_api_outlier(self, backend):
        # no torch.empty test because torch.empty generate a random tensor,
        # means nothing for sum ops

        pt_output = {
            'empty': torch.tensor(0.),
            'inf': torch.tensor(float('inf')),
            'nan': torch.tensor(float('nan'))
        }

        if backend == 'torch':
            x = torch.Tensor().npu()
            y = x.sum()
            print('torch sum empty: ', y)

            x = torch.empty(2, 3).fill_(float('inf')).npu()
            y = x.sum()
            print('torch sum inf: ', y)

            x = torch.empty(2, 3).fill_(float('nan')).npu()
            y = x.sum()
            print('torch sum nan: ', y)

        elif backend == 'mindspore':
            x = torch.Tensor([]).npu()
            y = x.sum()
            print('ms sum empty: ', y, ' torch sum empty: ', pt_output['inf'])
            assert torch.equal(y, pt_output['empty'])

            x = torch.empty(2, 3).fill_(float('inf'))
            y = x.sum()
            print('ms sum inf: ', y, ' torch sum inf: ', pt_output['inf'])
            assert torch.isinf(y).all()

            x = torch.empty(2, 3).fill_(float('nan'))
            y = x.sum()
            print('ms sum nan: ', y, ' torch sum nan: ', pt_output['nan'])
            assert torch.isnan(y).all()

    def test_performance(self, backend, repeat_times=1000, shape=(100, 100)):

        forward_cost_time = []
        backward_cost_time = []
        max_memory = []

        pt_forward_time_diff_shape = {
            (10, 10): 8.8076,
            (100, 100): 1.1525,
            (1000, 100): 1.2447,
            (1000, 1000): 1.2732
        }

        pt_backward_time_diff_shape = {
            (10, 10): 0.5485,
            (100, 100): 0.2642,
            (1000, 100): 1.9557,
            (1000, 1000): 2.6229
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
                y = x.sum().npu()
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
                y = x.sum()
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
            if forward_cost_time > pt_forward_time_diff_shape[shape] * 0.52:
                raise ValueError(
                    f'Expect ms forward cost time <= (pt cost time * 0.58), but got '
                    f'ms {forward_cost_time:.4f}ms, '
                    f'and pt {pt_forward_time_diff_shape[shape]:.4f}ms.'
                )

            # check backwward performence
            if backward_cost_time > pt_backward_time_diff_shape[shape] * 1.97:
                raise ValueError(
                    f'Expect ms backward cost time <= (pt cost time * 1.97), but got '
                    f'ms {backward_cost_time:.4f}ms, '
                    f'and pt {pt_backward_time_diff_shape[shape]:.4f}ms.'
                )

            # check memory usage
            if max_memory > pt_max_memory[shape] * 0.37:
                raise ValueError(f'Expect ms max memory <= (pt max memory * 0.37), but got '
                                 f'ms max memory {max_memory} Btye, and pt max memory {pt_max_memory[shape]} Btye.')

    def test_precision_pt(self, dtype=torch.bfloat16):
        x_grad = []

        def x_hook(grad):
            x_grad.append(grad)

        x = torch.randn((100, 100), dtype=dtype, requires_grad=True).npu()
        x.register_hook(x_hook)

        # forward and backward
        y = x.sum()
        y.backward()
        print(f"output is {y}")
        print(f"x.grad is {x_grad[0]}")

        # save inputs, outputs and grad
        save_dict = {
            'x': x.contiguous(),
            'y': y.contiguous(),
            'x_grad': x_grad[0].contiguous()
        }

        torch.save(save_dict, os.path.join(pt_input_output_path, f"input_output_sum_{dtype_str_dict[dtype]}.pt"))

    def test_precision_ms(self, dtype=torch.bfloat16):
        # load from saved data
        pt_data_path = os.path.join(pt_input_output_path, f"input_output_sum_{dtype_str_dict[dtype]}.pt")
        pt_data = torch.load(pt_data_path, map_location="cpu")

        x_pt, y_pt, x_grad_pt = pt_data['x'], pt_data['y'], pt_data['x_grad']

        weight_grad = []

        def weight_hook(grad):
            weight_grad.append(grad)

        x_pt.register_hook(weight_hook)
        # forward
        y_ms = x_pt.sum()

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
    # TestStd has some bugs to be fixed
    TestList = [TestAll, TestAny, TestArgmax, TestMax, TestMin, TestMean, TestNorm, TestSum, TestProd]
    test_precision_dtype = [torch.bfloat16, torch.float16]
    for TestCase in TestList:
        print(f"running test {TestCase.__name__} {args.test_mode}")
        test_case = TestCase()
        if args.test_mode == 'completeness':
            test_case.test_api_completeness()
        elif args.test_mode == 'performance':
            for i, shape in enumerate(test_shape):
                test_case.test_performance(args.backend, repeat_times[i], test_shape[i])
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
