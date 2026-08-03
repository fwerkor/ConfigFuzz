import os
import random
import numpy as np
import torch
import torch_npu


def seed_all(seed=1921):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['HCCL_DETERMINISTIC'] = 'true'
    os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
    os.environ['NCCL_DETERMINISTIC'] = '1'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)
