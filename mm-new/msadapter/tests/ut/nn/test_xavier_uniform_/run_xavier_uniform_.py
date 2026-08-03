import argparse
import os
import time
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import random
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)


class TestXavierUniform_(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()
        self.test_api_completeness_3()
        self.test_api_completeness_4()
        self.test_api_completeness_5()

    def test_api_completeness_1(self):
        gain = nn.init.calculate_gain('relu')
        w = torch.empty(3, 5)
        nn.init.xavier_uniform_(w, gain=gain)

    def test_api_completeness_2(self):
        gain = nn.init.calculate_gain('linear')
        w = torch.empty(3, 5)
        nn.init.xavier_uniform_(w, gain=gain)

    def test_api_completeness_3(self):
        gain = nn.init.calculate_gain('tanh')
        w = torch.empty(3, 5)
        nn.init.xavier_uniform_(w, gain=gain)

    def test_api_completeness_4(self):
        gain = nn.init.calculate_gain('leaky_relu')
        w = torch.empty(3, 5)
        nn.init.xavier_uniform_(w, gain=gain)

    def test_api_completeness_5(self):
        gain = nn.init.calculate_gain('selu')
        w = torch.empty(3, 5)
        nn.init.xavier_uniform_(w, gain=gain)

    def test_performance(self, backend):
        repeat_times = 100

        gain = nn.init.calculate_gain('relu')
        w = Parameter(torch.empty(3, 5))
        w.requires_grad_(True)

        start_time = time.time()
        for _ in range(repeat_times):
            nn.init.xavier_uniform_(w, gain=gain)

        cost_time = time.time() - start_time
        print(f'repeat {repeat_times} times cost time {cost_time}s')

        if backend == 'torch':
            if not os.path.exists('.tmp'):
                os.mkdir('.tmp')
            torch.save(cost_time, f'.tmp/cost_time.pt')

        elif backend == 'mindspore':
            pt_cost_time = 4.4066
            if cost_time > pt_cost_time / 0.8:
                raise ValueError(f'Expect ms cost time <= (pt cost time / 0.8), but got '
                                 f'ms cost time {cost_time}, and pt cost time {pt_cost_time}.')


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")

    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance'], help='test mode')
    parser.add_argument('--backend', type=str, choices=['torch', 'mindspore'], help='backend')
    args, _ = parser.parse_known_args()
    test_case = TestXavierUniform_()

    if args.test_mode == 'completeness':
        test_case.test_api_completeness()
    elif args.test_mode == 'performance':
        test_case.test_performance(args.backend)
