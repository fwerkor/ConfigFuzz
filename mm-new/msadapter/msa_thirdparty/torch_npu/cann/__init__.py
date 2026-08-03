import os
import mindspore
from mindspore import ops

import torch

class CannOpBuilder():
    def __init__(self, ):
        current_dir = os.path.dirname(__file__)
        csrc_root = f"{current_dir}/../../../csrc"
        self.csrc_dir = f"{csrc_root}/cann_ops"
        self.cann_op = {}

    def load(self, csrc_name):
        src = [f"{self.csrc_dir}/{csrc_name}.cpp"]
        cann_builder = ops.CustomOpBuilder(csrc_name, src, "Ascend")
        self.cann_op[csrc_name] = cann_builder.load()

        return self.cann_op[csrc_name]

builder = CannOpBuilder()

def npu_apply_fused_ema_adamw(grad: torch.Tensor,
                              var: torch.Tensor,
                              m: torch.Tensor,
                              v: torch.Tensor,
                              s: torch.Tensor,
                              step: torch.Tensor,
                              lr: float = 1e-3,
                              ema_decay: float = 0.9999,
                              beta1: float = 0.9,
                              beta2: float = 0.999,
                              eps: float = 1e-8,
                              mode: int = 1,
                              bias_correction: bool = True,
                              weight_decay: float = 0.0):
    return builder.load("npu_apply_fused_ema_adam").npu_apply_fused_ema_adamw(grad,
                                                                              var,
                                                                              m,
                                                                              v,
                                                                              s,
                                                                              step,
                                                                              lr,
                                                                              ema_decay,
                                                                              beta1,
                                                                              beta2,
                                                                              eps,
                                                                              mode,
                                                                              bias_correction,
                                                                              weight_decay)
