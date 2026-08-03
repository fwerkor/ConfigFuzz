import os
import time
import random
import argparse

import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

import numpy as np


class TestCrossEntropyLoss(TestCase):

    def test_api_completeness(self):
        # torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
        #                           reduce=None, reduction='mean', label_smoothing=0.0)
        logits = torch.randn(3, 5)
        target = torch.randint(0, 5, (3,), dtype=torch.long)

        loss_fn = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                      reduce=None, reduction='mean', label_smoothing=0.0)
        loss = loss_fn(input=logits, target=target)
        print(loss)

        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fn(input=logits, target=target)
        print(loss)

        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.5)
        loss = loss_fn(input=logits, target=target)
        print(loss)


    def test_precision_pt(self, dtype=torch.bfloat16):

        loss_fn = nn.CrossEntropyLoss().npu().train()

        logits = torch.randn((128, 256)).to(dtype).npu()
        logits.requires_grad_(True)
        target = torch.randint(0, 256, (128,), dtype=torch.long).npu()

        # forward and backward
        loss = loss_fn(logits, target)
        loss.backward()
        print(f"loss is {loss}")

        # save inputs, outputs and grad
        save_dict = {
            'logits': logits,
            'target': target,
            'loss': loss,
            'logits_grad': logits.grad
        }

        input_path = "/home/workspace/mindspore_dataset/msadapter/test_input/ut"
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        if dtype == torch.bfloat16:
            data_name = "CrossEntropyLossData_bf16.pt"
        elif dtype == torch.float16:
            data_name = "CrossEntropyLossData_fp16.pt"
        else:
            data_name = "CrossEntropyLossData_fp32.pt"  # torch.float32
        torch.save(save_dict, os.path.join(input_path, data_name))


    def test_precision_ms(self, dtype='bfloat16'):
        from mindspore.ops import composite as C

        logits_grad = []
        def logits_hook(grad):
            print(f"grad is {grad}")
            logits_grad.append(grad)

        loss_fn = nn.CrossEntropyLoss()

        # load from saved data
        input_path = "/home/workspace/mindspore_dataset/msadapter/test_input/ut"
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        if dtype == 'bfloat16':
            data_name = "CrossEntropyLossData_bf16.pt"
        elif dtype == 'float16':
            data_name = "CrossEntropyLossData_fp16.pt"
        else:
            data_name = "CrossEntropyLossData_fp32.pt" # float32

        pt_data = torch.load(os.path.join(input_path, data_name), map_location="cpu")
        # logits, target = torch.nn.parameter.Parameter(pt_data['logits']), pt_data['target']
        logits, target = pt_data['logits'], pt_data['target']
        logits.register_hook(logits_hook)
        pt_loss, pt_logits_grad = pt_data['loss'], pt_data['logits_grad']

        # forward
        def flag_func():
            pass
        loss = loss_fn(logits, target)
        # backward
        loss.backward()
        # check forward precision
        print(f"ms loss is {loss}, pt loss is {pt_loss}")
        loss_ms_np = loss.to(torch.float32).numpy()
        loss_pt_np = pt_loss.to(torch.float32).numpy()
        assert np.allclose(loss_ms_np, loss_pt_np)

        # check backward precision
        assert logits_grad and logits_grad[0] is not None, f"Expect logits_grad is not None, but got {logits_grad}"
        print(f"ms grad is:\n{logits_grad[0]},\npt grad is:\n{pt_logits_grad}")
        grad_ms_np = logits_grad[0].to(torch.float32).numpy()
        grad_pt_np = pt_logits_grad.to(torch.float32).numpy()
        assert np.allclose(grad_ms_np, grad_pt_np)


    def test_performance(self, backend):

        repeat_times = 100

        loss_fn = nn.CrossEntropyLoss()

        logits = torch.randn((128, 256)).npu()
        logits.requires_grad_(True).npu()
        target = torch.randint(0, 256, (128,), dtype=torch.long).npu()

        loss = loss_fn(logits, target)  # pre-build

        start_time = time.time()
        for _ in range(repeat_times):
            loss = loss_fn(logits, target)
            print(loss)
        cost_time = time.time() - start_time
        print(f"repeat {repeat_times} times cost time {cost_time}s")

        if backend == "torch":
            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")
            torch.save(cost_time, f".tmp/cost_time.pt")
        elif backend == "mindspore":
            # pt_cost_time = torch.load(f".tmp/cost_time.pt", map_location="cpu")
            pt_cost_time = 0.228788
            if cost_time > pt_cost_time / 0.8:
                raise ValueError(f"Expect ms cost time <= (pt cost time / 0.8), but got "
                                 f"ms cost time {cost_time}, and pt cost time {pt_cost_time}.")


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
    # set seed

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'precision', 'performance'],
                        help="test mode")
    parser.add_argument('--backend', type=str, choices=['torch', 'mindspore'],
                        help="backend")
    parser.add_argument('--dtype', type=str, choices=['bfloat16', 'float16', 'float32'],
                        help="dtype")
    args, _ = parser.parse_known_args()

    test_case = TestCrossEntropyLoss()
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    if args.test_mode == 'completeness':
        test_case.test_api_completeness()
    elif args.test_mode == 'precision':
        if args.backend == 'torch':
            test_case.test_precision_pt(dtype_map[args.dtype])
        elif args.backend == 'mindspore':
            test_case.test_precision_ms(args.dtype)
    elif args.test_mode == 'performance':
        test_case.test_performance(args.backend)
