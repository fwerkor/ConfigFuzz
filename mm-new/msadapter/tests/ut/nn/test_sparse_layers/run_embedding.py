import os
import time
import random
import argparse

import torch
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

import numpy as np
from pathlib import Path


class TestEmbedding(TestCase):

    def test_api_completeness(self):
        # pta: torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
        # scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
        # ms: Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.,
        # scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, dtype=None)

        # keyword device not supported

        inputs = torch.randint(0, 256, (32, 512))

        embedding  = nn.Embedding(256, 16)
        output = embedding(inputs)
        print(output)

        embedding  = nn.Embedding(256, 16, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False,
                                  sparse=False, _weight=None, _freeze=False, dtype=None)
        output = embedding(inputs)
        print(output)

        weight = torch.randn(256, 16, requires_grad=True).npu()
        embedding = nn.Embedding.from_pretrained(weight, padding_idx=0, max_norm=1., norm_type=1., scale_grad_by_freq=True,
                                  sparse=True)
        output = embedding(inputs)
        print(output)

        # pta: from_pretrained(embeddings, freeze=True, padding_idx=None, max_norm=None, norm_type=2.0,
        # scale_grad_by_freq=False, sparse=False)
        # ms: from_pretrained(cls, embeddings, freeze=True, padding_idx=None, max_norm=None, norm_type=2.,
        # scale_grad_by_freq=False, sparse=False)
        weight = torch.randn(256, 16)
        embedding = nn.Embedding.from_pretrained(weight)
        output = embedding(inputs)
        print(output)


    def test_precision_pt(self):

        weight = torch.randn(256, 16, requires_grad=True)
        embedding = nn.Embedding.from_pretrained(weight, freeze=False).npu().train()

        inputs = torch.randint(0, 256, (32, 512)).to(torch.long).npu()

        # forward and backward
        output = embedding(inputs)
        output = output.mean()
        output.backward()
        print(f"output is {output}")

        # save inputs, outputs and grad
        save_dict = {
            'inputs': inputs,
            'weight': weight,
            'weight_grad': embedding.weight.grad,
            'output': output,
            # 'output_grad': output.grad
        }
        dir_path = Path.cwd()
        tests_path = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))
        test_input = "/home/workspace/mindspore_dataset/msadapter/test_input"
        test_path = os.path.join(test_input, os.path.relpath(dir_path, tests_path))

        if not os.path.exists(test_path):
            os.makedirs(test_path)
        torch.save(save_dict, os.path.join(test_path, "EmbeddingData.pt"))


    def test_precision_ms(self):
        from mindspore.ops import composite as C
        from mindspore.common.api import _pynative_executor

        weight_grad = []
        def weight_hook(grad):
            print(f"grad is {grad}")
            weight_grad.append(grad)

        dir_path = Path.cwd()
        tests_path = os.path.dirname(os.path.dirname(os.path.dirname(dir_path)))
        test_input = "/home/workspace/mindspore_dataset/msadapter/test_input"
        test_path = os.path.join(test_input, os.path.relpath(dir_path, tests_path))

        # load from saved data
        pt_data = torch.load(os.path.join(test_path, "EmbeddingData.pt"), map_location="cpu")
        inputs, weight = pt_data['inputs'], pt_data['weight']
        pt_output, pt_weight_grad = pt_data['output'], pt_data['weight_grad']

        print(f"pt grad is:\n{pt_weight_grad}")

        embedding = nn.Embedding.from_pretrained(weight, freeze=False).npu()
        embedding.weight.register_hook(weight_hook)

        # forward
        def flag_func():
            pass
        output = embedding(inputs)
        output = output.mean()
        output.backward()

        # check forward precision
        print(f"ms output is {output}, pt output is {pt_output}")
        if np.allclose(output, pt_output, 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="

        # check backward precision
        assert weight_grad and weight_grad[0] is not None, f"Expect weight_grad is not None, but got {weight_grad}"
        print(f"ms grad is:\n{weight_grad[0]},\npt grad is:\n{pt_weight_grad}")
        if np.allclose(weight_grad[0], pt_weight_grad, 0.0, 0.0) == False:
            assert False, "=============== Accuracy test fail !!! ==============="


    def test_performance(self, backend):

        repeat_times = 100

        weight = torch.randn(256, 16, requires_grad=True)
        embedding = nn.Embedding.from_pretrained(weight, freeze=False).train()

        inputs = torch.randint(0, 256, (32, 512)).to(torch.long).npu()

        # forward and backward
        output = embedding(inputs)

        start_time = time.time()
        for _ in range(repeat_times):
            output = embedding(inputs)
        cost_time = time.time() - start_time
        print(f"repeat {repeat_times} times cost time {cost_time}s")

        # not support for torch
        if backend == "torch":
            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")
            torch.save(cost_time, f".tmp/cost_time.pt")
        elif backend == "mindspore":
            # pt_cost_time = torch.load(f".tmp/cost_time.pt", map_location="cpu")
            pt_cost_time = 0.04276
            print(f"ms_time / pt_time = {cost_time} / {pt_cost_time} = {cost_time / pt_cost_time}")
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
    args, _ = parser.parse_known_args()

    test_case = TestEmbedding()

    if args.test_mode == 'completeness':
        test_case.test_api_completeness()
    elif args.test_mode == 'precision':
        if args.backend == 'torch':
            test_case.test_precision_pt()
        elif args.backend == 'mindspore':
            test_case.test_precision_ms()
    elif args.test_mode == 'performance':
        test_case.test_performance(args.backend)
