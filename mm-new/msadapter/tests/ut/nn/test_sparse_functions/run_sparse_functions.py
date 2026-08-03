import os
import torch
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests


class TestSparseFunctions(TestCase):
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    def test_embedding_api_completeness(self):
        # pta: torch.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None,
        #                               norm_type=2.0, scale_grad_by_freq=False, sparse=False)

        # ms: mindspore.mint.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None,
        #                                            norm_type=2.0, scale_grad_by_freq=False)
        # keyword 'sparse' not supported

        inputs = torch.randint(0, 512, (32, 256))
        weight = torch.randn(256, 16)

        outputs = F.embedding(inputs, weight)
        print(outputs)

        outputs = F.embedding(inputs, weight, padding_idx=None, max_norm=None,
                              norm_type=2.0, scale_grad_by_freq=False)
        print(outputs)

        outputs = F.embedding(inputs, weight, padding_idx=0, max_norm=1.0,
                              norm_type=1.0, scale_grad_by_freq=True)
        print(outputs)

    def test_one_hot_api_completeness(self):
        # pta: torch.nn.functional.one_hot(tensor, num_classes=-1) → LongTensor

        # ms: one_hot(tensor, num_classes=-1)

        inputs = torch.randint(0, 128, (32, 256))

        outputs = F.one_hot(inputs)
        print(outputs)

        outputs = F.one_hot(tensor=inputs, num_classes=-1)
        print(outputs)

        outputs = F.one_hot(inputs, num_classes=130)
        print(outputs)


