import os
import time
import random
import argparse
import torch
import torch_npu
from torch.configs import MS271

from torch_npu.testing.testcase import TestCase

import numpy as np

test_input = "/home/workspace/mindspore_dataset/msadapter/test_input"
test_path = os.path.join(test_input, "ops", "ckpt")


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


class TestArange(TestCase):

    def test_api_completeness(self):
        start = 0
        end = 1
        step = 1
        dtype = None
        device = torch.device('cuda')
        y = torch.arange(start=start, end=end, step=step, dtype=dtype, device=device)

        start = 0
        end = 1
        step = 1
        dtype = torch.bfloat16
        y = torch.arange(start=start, end=end, step=step, dtype=dtype)

        start = 1
        end = 2.5
        step = 0.5
        dtype = torch.float32
        y = torch.arange(start=start, end=end, step=step, dtype=dtype)

        print("arange: ", y)
        benchmark = torch.tensor([1., 1.5, 2.], dtype=torch.float32)
        assert torch.equal(y, benchmark)

    def test_performance(self, backend):

        repeat_times = 100

        start = 1
        end = 2.5
        step = 0.5
        dtype = torch.bfloat16

        # forward and backward
        y = torch.arange(start=start, end=end, step=step, dtype=dtype).npu()

        start_time = time.time()
        for _ in range(repeat_times):
            y = torch.arange(start=start, end=end, step=step, dtype=dtype).npu()
            print(y)

        cost_time = time.time() - start_time
        print(f"repeat {repeat_times} times cost time {cost_time}s")

        # not support for torch
        if backend == "torch":
            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")
            torch.save(cost_time, f".tmp/cost_time.pt")

        elif backend == "mindspore":
            pt_cost_time = 0.04816
            if cost_time > pt_cost_time * 4.1:
                raise ValueError(f"Expect ms cost time <= (pt cost time * 4.1), but got "
                                 f"ms cost time {cost_time}, and pt cost time {pt_cost_time}.")

    def test_precision_pt(self):

        start = 1
        end = 2.5
        step = 0.5
        dtype = torch.float32

        # forward and backward
        y = torch.arange(start=start, end=end, step=step, dtype=dtype)

        # save inputs, outputs and grad
        save_dict = {
            'y': y,
        }

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        torch.save(save_dict, f".tmp/arange.pt")

    def test_precision_ms(self):

        # load from saved data
        pt_data = torch.load(os.path.join(test_path, "arange.pt"), map_location="cpu")

        y_pt = torch.tensor([1., 1.5, 2.])

        start = 1
        end = 2.5
        step = 0.5
        dtype = torch.float32

        y_ms = torch.arange(start=start, end=end, step=step, dtype=dtype)

        # check forward precision
        print(f"ms y is {y_ms}, pt y is {y_pt}")
        if np.allclose(y_ms, y_pt, 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestLinspace(TestCase):

    def test_api_completeness(self):
        start = 3
        end = 10
        steps = 5
        dtype = None
        y = torch.linspace(start=start, end=end, steps=steps, dtype=dtype)

        start = -10
        end = 10
        steps = 5
        dtype = torch.bfloat16
        y = torch.linspace(start=start, end=end, steps=steps, dtype=dtype)

        start = -10
        end = 10
        steps = 1
        dtype = torch.float32
        y = torch.linspace(start=start, end=end, steps=steps, dtype=dtype)

        print("linspace: ", y)
        benchmark = torch.tensor([-10.], dtype=torch.float32)
        assert torch.equal(y, benchmark)

    def test_performance(self, backend):

        repeat_times = 100

        start = -10
        end = 10
        steps = 5
        dtype = torch.bfloat16

        # forward
        y = torch.linspace(start=start, end=end, steps=steps, dtype=dtype).npu()

        start_time = time.time()
        for _ in range(repeat_times):
            y = torch.linspace(start=start, end=end, steps=steps, dtype=dtype).npu()
            print(y)

        cost_time = time.time() - start_time
        print(f"repeat {repeat_times} times cost time {cost_time}s")

        # not support for torch
        if backend == "torch":
            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")
            torch.save(cost_time, f".tmp/cost_time.pt")

        elif backend == "mindspore":
            pt_cost_time = 0.15356
            if cost_time > pt_cost_time * 0.4:
                raise ValueError(f"Expect ms cost time <= (pt cost time * 0.4), but got "
                                 f"ms cost time {cost_time}, and pt cost time {pt_cost_time}.")

    def test_precision_pt(self):

        start = -10
        end = 10
        steps = 5
        dtype = torch.float32

        # forward
        y = torch.linspace(start=start, end=end, steps=steps, dtype=dtype)
        save_dict = {
            'y': y,
        }

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        torch.save(save_dict, f".tmp/linspace.pt")

    def test_precision_ms(self):

        # load from saved data
        pt_data = torch.load(os.path.join(test_path, "linspace.pt"), map_location="cpu")

        y_pt = pt_data['y']

        start = -10
        end = 10
        steps = 5
        dtype = torch.float32

        y_ms = torch.linspace(start=start, end=end, steps=steps, dtype=dtype)

        # check forward precision
        print(f"ms y is {y_ms}, pt y is {y_pt}")
        if np.allclose(y_ms, y_pt, 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestOnes(TestCase):

    def test_api_completeness(self):
        size = (2, 3)
        dtype = None
        device = torch.device('cuda')
        y = torch.ones(size, dtype=dtype, device=device)

        size = 3
        dtype = torch.float32
        y = torch.ones(size, dtype=dtype)

        size = 5
        dtype = torch.bfloat16
        y = torch.ones(size, dtype=dtype)

        print("ones: ", y)
        benchmark = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.bfloat16)
        assert torch.equal(y, benchmark)

    def test_performance(self, backend):

        repeat_times = 100

        size = 5
        dtype = torch.bfloat16

        # forward and backward
        y = torch.ones(size, dtype=dtype).npu()

        start_time = time.time()
        for _ in range(repeat_times):
            y = torch.ones(size, dtype=dtype).npu()
            print(y)

        cost_time = time.time() - start_time
        print(f"repeat {repeat_times} times cost time {cost_time}s")

        # not support for torch
        if backend == "torch":
            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")
            torch.save(cost_time, f".tmp/cost_time.pt")

        elif backend == "mindspore":
            pt_cost_time = 0.1599
            if cost_time > pt_cost_time * 0.5:
                raise ValueError(f"Expect ms cost time <= (pt cost time * 0.5), but got "
                                 f"ms cost time {cost_time}, and pt cost time {pt_cost_time}.")

    def test_precision_pt(self):

        size = 5
        dtype = torch.float32

        # forward
        y = torch.ones(size, dtype=dtype)

        save_dict = {
            'y': y,
        }

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        torch.save(save_dict, f".tmp/ones.pt")

    def test_precision_ms(self):

        # load from saved data
        pt_data = torch.load(os.path.join(test_path, "ones.pt"), map_location="cpu")

        y_pt = pt_data['y']

        size = 5
        dtype = torch.float32

        y_ms = torch.ones(size, dtype=dtype)

        # check forward precision
        print(f"ms y is {y_ms}, pt y is {y_pt}")
        if np.allclose(y_ms, y_pt, 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestOneslike(TestCase):

    def test_api_completeness(self):
        x = torch.empty(2, 3)
        dtype = None
        device = torch.device('cuda')
        y = torch.ones_like(x, dtype=dtype, device=device)

        x = torch.empty(2, 3)
        dtype = torch.float32
        y = torch.ones_like(x, dtype=dtype)

        x = torch.empty(2, 3)
        dtype = torch.bfloat16
        y = torch.ones_like(x, dtype=dtype)

        print("oneslike: ", y)
        benchmark = torch.tensor([[1., 1., 1.], [1., 1., 1.]], dtype=torch.bfloat16)
        assert torch.equal(y, benchmark)

    def test_performance(self, backend):

        repeat_times = 100

        x = torch.empty(2, 3).npu()
        dtype = torch.bfloat16

        # forward
        y = torch.ones_like(x, dtype=dtype).npu()

        start_time = time.time()
        for _ in range(repeat_times):
            y = torch.ones_like(x, dtype=dtype).npu()
            print(y)

        cost_time = time.time() - start_time
        print(f"repeat {repeat_times} times cost time {cost_time}s")

        # not support for torch
        if backend == "torch":
            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")
            torch.save(cost_time, f".tmp/cost_time.pt")

        elif backend == "mindspore":
            pt_cost_time = 0.179
            if cost_time > pt_cost_time * 0.4:
                raise ValueError(f"Expect ms cost time <= (pt cost time * 0.4), but got "
                                 f"ms cost time {cost_time}, and pt cost time {pt_cost_time}.")

    def test_precision_pt(self):

        x = torch.empty(2, 3).npu()
        dtype = torch.float32

        # forward
        y = torch.ones_like(x, dtype=dtype)

        save_dict = {
            'y': y,
        }

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        torch.save(save_dict, f".tmp/ones_like.pt")

    def test_precision_ms(self):

        # load from saved data
        pt_data = torch.load(os.path.join(test_path, "ones_like.pt"), map_location="cpu")

        y_pt = pt_data['y']

        x = torch.empty(2, 3).npu()
        dtype = torch.float32

        y_ms = torch.ones_like(x, dtype=dtype)

        # check forward precision
        print(f"ms y is {y_ms}, pt y is {y_pt}")
        if np.allclose(y_ms, y_pt, 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestZeros(TestCase):

    def test_api_completeness(self):
        size = (2, 3)
        dtype = None
        device = torch.device('cuda')
        layout = torch.strided
        requires_grad = False
        y = torch.zeros(size, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)

        size = 3
        dtype = torch.float32
        y = torch.zeros(size, dtype=dtype)

        size = 5
        dtype = torch.bfloat16
        requires_grad = True
        y = torch.zeros(size, dtype=dtype, requires_grad=requires_grad)

        print("zeros: ", y)
        benchmark = torch.tensor([0., 0., 0., 0., 0.], dtype=torch.bfloat16)
        assert torch.equal(y, benchmark)

    def test_performance(self, backend):

        repeat_times = 100

        size = 5
        dtype = torch.bfloat16
        requires_grad = True

        # forward
        y = torch.zeros(size, dtype=dtype, requires_grad=requires_grad).npu()

        start_time = time.time()
        for _ in range(repeat_times):
            y = torch.zeros(size, dtype=dtype, requires_grad=requires_grad).npu()
            print(y)

        cost_time = time.time() - start_time
        print(f"repeat {repeat_times} times cost time {cost_time}s")

        # not support for torch
        if backend == "torch":
            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")
            torch.save(cost_time, f".tmp/cost_time.pt")

        elif backend == "mindspore":
            pt_cost_time = 0.06341
            if cost_time > pt_cost_time * 1:
                raise ValueError(f"Expect ms cost time <= (pt cost time * 1), but got "
                                 f"ms cost time {cost_time}, and pt cost time {pt_cost_time}.")

    def test_precision_pt(self):

        size = 5
        dtype = torch.float32
        requires_grad = True

        # forward
        y = torch.zeros(size, dtype=dtype, requires_grad=requires_grad)

        save_dict = {
            'y': y,
        }

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        torch.save(save_dict, f".tmp/zeros.pt")

    def test_precision_ms(self):

        # load from saved data
        pt_data = torch.load(os.path.join(test_path, "zeros.pt"), map_location="cpu")

        y_pt = pt_data['y']

        size = 5
        requires_grad = True
        dtype = torch.float32

        y_ms = torch.zeros(size, dtype=dtype, requires_grad=requires_grad)

        # check forward precision
        print(f"ms y is {y_ms}, pt y is {y_pt}")
        if np.allclose(y_ms, y_pt, 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestZeroslike(TestCase):

    def test_api_completeness(self):
        x = torch.empty(2, 3)
        dtype = None
        y = torch.zeros_like(x, dtype=dtype, memory_format=torch.preserve_format)

        x = torch.empty(2, 3)
        dtype = torch.float32
        y = torch.zeros_like(x, dtype=dtype)

        x = torch.empty(2, 3)
        dtype = torch.bfloat16
        y = torch.zeros_like(x, dtype=dtype)

        print("zeroslike: ", y)
        benchmark = torch.tensor([[0., 0., 0.], [0., 0., 0.]], dtype=torch.bfloat16)
        assert torch.equal(y, benchmark)

    def test_performance(self, backend):

        repeat_times = 100

        x = torch.empty(2, 3).npu()
        dtype = torch.bfloat16

        # forward
        y = torch.zeros_like(x, dtype=dtype).npu()

        start_time = time.time()
        for _ in range(repeat_times):
            y = torch.zeros_like(x, dtype=dtype).npu()
            print(y)

        cost_time = time.time() - start_time
        print(f"repeat {repeat_times} times cost time {cost_time}s")

        # not support for torch
        if backend == "torch":
            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")
            torch.save(cost_time, f".tmp/cost_time.pt")

        elif backend == "mindspore":
            pt_cost_time = 0.06769
            if cost_time > pt_cost_time * 1:
                raise ValueError(f"Expect ms cost time <= (pt cost time * 1), but got "
                                 f"ms cost time {cost_time}, and pt cost time {pt_cost_time}.")

    def test_precision_pt(self):

        x = torch.empty(2, 3).npu()
        dtype = torch.float32

        # forward
        y = torch.zeros_like(x, dtype=dtype)

        save_dict = {
            'y': y,
        }

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        torch.save(save_dict, f".tmp/zeros_like.pt")

    def test_precision_ms(self):

        # load from saved data
        pt_data = torch.load(os.path.join(test_path, "zeros_like.pt"), map_location="cpu")

        y_pt = pt_data['y']

        x = torch.empty(2, 3).npu()
        dtype = torch.float32

        y_ms = torch.zeros_like(x, dtype=dtype)

        # check forward precision
        print(f"ms y is {y_ms}, pt y is {y_pt}")
        if np.allclose(y_ms, y_pt, 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


class TestEmptyLike(TestCase):
    def test_api_completeness(self):
        if MS271:
            self.test_pin_memory()

    def test_pin_memory(self):
        input = torch.tensor(10.)
        x = torch.empty_like(input)
        assert not x.is_pinned()

        x = torch.empty_like(input, pin_memory=False)
        assert not x.is_pinned()

        x = torch.empty_like(input, device='cpu', pin_memory=True)
        assert x.is_pinned()

        x = torch.empty_like(input, device='cpu', pin_memory=False)
        assert not x.is_pinned()


class TestFrombuffer(TestCase):


    def test_api_completeness(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tensor = torch.frombuffer(np_arr.data, dtype=torch.float32)
        benchmark = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        assert torch.equal(tensor, benchmark), f"test frombuffer with dtype float32 error"

        np_empty_arr = np.array([], dtype=np.float32)
        tensor_empty = torch.frombuffer(np_empty_arr.data, dtype=torch.float32)
        assert tensor_empty.numel() == 0, f"test frombuffer with empty data error"


    def test_performance(self, backend):
        repeat_times = 100

        sample_data = torch.arange(0, 10, dtype=torch.float32)
        buffer = sample_data.numpy().tobytes() 
        dtype = torch.float32 

        start_time = time.time()
        for _ in range(repeat_times):
            x = torch.frombuffer(buffer, dtype=dtype)
            print(x)

        cost_time = time.time() - start_time
        print(f"repeat {repeat_times} times cost time {cost_time}s")

        if backend == "torch":
            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")
            torch.save(cost_time, f".tmp/frombuffer_cost_time.pt")

        elif backend == "mindspore":
            pt_cost_time = 0.04 
            if cost_time > pt_cost_time * 1.5:
                raise ValueError(f"Expect ms cost time <= (pt cost time * 1.2), but got "
                                f"ms cost time {cost_time}, and pt cost time {pt_cost_time}.")
        
    def test_precision_pt(self):

        dtype = torch.float32
        np_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = torch.frombuffer(np_arr.data, dtype=torch.float32)

        save_dict = {
            'y': y,
        }

        if not os.path.exists(".tmp"):
            os.mkdir(".tmp")

        torch.save(save_dict, f".tmp/frombuffer.pt")


    def test_precision_ms(self):


        pt_data = torch.load(os.path.join(test_path, "frombuffer.pt"), map_location="cpu")
        
        y_pt = pt_data['y']

        np_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        dtype = torch.float32

        y_ms = torch.frombuffer(np_arr.data, dtype=torch.float32)

        print(f"ms y is {y_ms}, pt y is {y_pt}")
        if np.allclose(y_ms, y_pt, 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    parser.add_argument('--backend', type=str, choices=['torch', 'mindspore'],
                        help="backend")

    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestArange().test_api_completeness()
        TestLinspace().test_api_completeness()
        TestOnes().test_api_completeness()
        TestOneslike().test_api_completeness()
        TestZeros().test_api_completeness()
        TestZeroslike().test_api_completeness()
        TestEmptyLike().test_api_completeness()

    elif args.test_mode == 'precision':

        if args.backend == 'torch':
            TestArange().test_precision_pt()
            TestLinspace().test_precision_pt()
            TestOnes().test_precision_pt()
            TestOneslike().test_precision_pt()
            TestZeros().test_precision_pt()
            TestZeroslike().test_precision_pt()

        elif args.backend == 'mindspore':
            TestArange().test_precision_ms()
            TestLinspace().test_precision_ms()
            TestOnes().test_precision_ms()
            TestOneslike().test_precision_ms()
            TestZeros().test_precision_ms()
            TestZeroslike().test_precision_ms()

    elif args.test_mode == 'performance':
        TestArange().test_performance(args.backend)
        TestLinspace().test_performance(args.backend)
        TestOnes().test_performance(args.backend)
        TestOneslike().test_performance(args.backend)
        TestZeros().test_performance(args.backend)
        TestZeroslike().test_performance(args.backend)
