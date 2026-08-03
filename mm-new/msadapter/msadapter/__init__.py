# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""core module"""
__version__ = "0.6.0"

import os
import sys
import importlib.metadata
from typing import Optional, Union

def _running_with_deploy():
    return False

import mindspore
from mindspore import context
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error

os.environ['RANK'] = os.getenv('RANK_ID', '0')
os.environ['WORLD_SIZE'] = os.getenv('MS_WORKER_NUM', '1')

if 'RANK_TABLE_FILE' in os.environ:
    del os.environ['RANK_TABLE_FILE']

channels_last = None
memory_format = None
preserve_format = None
legacy_contiguous_format = contiguous_format = None

from mindspore.runtime import Stream
from mindspore.ops.function import nn_func
from mindspore.ops import auto_generate as gen
from mindspore._c_expression import run_backward
from mindspore._c_expression.typing import Type as dtype
from mindspore.ops.operations._sequence_ops import TensorToScalar
from mindspore.ops.operations._inner_ops import Generator as GeneratorOp
from mindspore import Tensor, default_generator, Generator as msGenerator
from mindspore.common.api import _pynative_executor
from msadapter.configs import MS28

# Patch: auto-convert numpy.ndarray indices to list for MindSpore Tensor __getitem__
import numpy as _np

def _convert_ms_tensor_index(idx):
    if isinstance(idx, _np.ndarray):
        return idx.tolist()
    if isinstance(idx, tuple):
        return tuple(_convert_ms_tensor_index(i) for i in idx)
    if isinstance(idx, list):
        return [_convert_ms_tensor_index(i) for i in idx]
    return idx

_orig_tensor_getitem = Tensor.__getitem__
def _patched_tensor_getitem(self, idx):
    idx = _convert_ms_tensor_index(idx)
    return _orig_tensor_getitem(self, idx)

Tensor.__getitem__ = _patched_tensor_getitem
_pynative_executor.set_grad_flag(True)
from .proxy import enable_torch_proxy
enable_torch_proxy()

import msadapter
from msadapter._dtypes import *
from msadapter._layout import *
from msadapter.linalg import *
from msadapter._C.size import Size
from msadapter import _tensor
from msadapter.types import Device, device, UntypedStorage
from msadapter import overrides

class Generator(msGenerator):
    def __init__(self, device='cpu'):
        self._seed = msadapter.nn.Parameter(Tensor(0, mindspore.int64), requires_grad=False)
        self._offset = msadapter.nn.Parameter(Tensor(0, mindspore.int64), requires_grad=False)
        try:
            self._generator = GeneratorOp(self._seed, self._offset).set_device("CPU")
        except TypeError:
            self._generator = GeneratorOp().set_device("CPU")
        self._generator.add_prim_attr("manual_seed", False)
        self._to_scalar = TensorToScalar()

dtype = Type
Event = None
_tensor_classes: set[type["msadapter.Tensor"]] = set()

from msadapter.ops import *
from msadapter._C.typed_tensor import *
from msadapter._C import is_grad_enabled
from msadapter.serialization import load, save
from msadapter.amp import autocast, GradScaler
from msadapter import amp, random, serialization,  utils, jit
from msadapter.random import get_rng_state, initial_seed, manual_seed, seed, set_rng_state
from msadapter import optim, ops, nn, distributions, cuda, distributed, multiprocessing
from msadapter.autograd import no_grad, enable_grad, inference_mode, set_grad_enabled
from msadapter._bind import get_default_dtype, set_default_dtype, iinfo, finfo
from msadapter import backends, _dynamo, _inductor, _jit_internal, profiler, version, fx
from msadapter.nn.functional import log_softmax, layer_norm
from msadapter import compiler

long = int64
int = int32
float = float32
bool = bool_
cfloat = complex64
cdouble = complex128

npu = cuda
Ascend = cuda
cpu = cuda

AUTO_CAST_DTYE = {
    'cuda': msadapter.bfloat16,
    'cpu': msadapter.bfloat16,
    'npu': msadapter.float16,
    'Ascend': msadapter.float16
}

def tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)


def relu_(data: msadapter.Tensor):
    return nn_func.relu_(data)

def is_same_size(tensor1: Tensor, tensor2: Tensor) -> bool:
    return tensor1.size() == tensor2.size()

def use_deterministic_algorithms(deterministic: bool, warn_only: bool=False):
    mindspore.set_deterministic(deterministic)

def is_deterministic_algorithms_warn_only_enabled():
    return False

def are_deterministic_algorithms_enabled():
    return os.environ.get("TE_PARALLEL_COMPILER", 'false') == "1" and os.environ.get("HCCL_DETERMINISTIC", 'false') == "true"

def set_autocast_dtype(device_type, dtype):
    assert device_type in AUTO_CAST_DTYE.keys(), f'{device_type} is not in {AUTO_CAST_DTYE.keys()}'
    AUTO_CAST_DTYE[device_type] = dtype

def get_autocast_dtype(device_type):
    return AUTO_CAST_DTYE[device_type]

def is_autocast_enabled(*args, **kwargs):
    return False

def use_deterministic_algorithms(flag: bool):
    context.set_context(deterministic='ON' if flag else 'OFF')

def compile(fn=None, *args, **kwargs):
    def wrap_func(fn):
        return fn
    if fn is not None:
        return wrap_func(fn)
    return wrap_func

def inplace_apply_func(tensor_list, inv_scale):
    for idx, _ in enumerate(tensor_list):
        tensor_list[idx].copy_(ops.mul(tensor_list[idx], inv_scale))

def _amp_foreach_non_finite_check_and_unscale_(main_grads, found_inf, inv_scale):
    inplace_apply_func(main_grads, inv_scale)
    for grad in main_grads:
        found_inf = ops.logical_and(found_inf, ops.logical_not(ops.isfinite(grad)).all())

def get_num_threads():
    return os.cpu_count()

def set_num_threads(thread_num):
    pass

def svd_lowrank():
    pass

def _register_device_module(device_type, module):
    r"""Register an external runtime module of the specific :attr:`device_type`
    supported by msadapter.
    After the :attr:`module` is registered correctly, the user can refer
    the external runtime module as part of torch with attribute msadapter.xxx.
    """
    # Make sure the device_type represent a supported device type for msadapter.
    # device_type = msadapter.device(device_type)
    m = sys.modules[__name__]
    if hasattr(m, device_type):
        raise RuntimeError(f"The runtime module of '{device_type}' has already been registered with '{getattr(m, device_type)}'")
    setattr(m, device_type, module)
    torch_module_name = '.'.join([__name__, device_type])
    sys.modules[torch_module_name] = module

def set_printoptions(precision):
    pass

def get_device(tensor: msadapter.Tensor = None):
    return mindspore.get_current_device().device_id

def get_device_module(device: Optional[Union[Device, str]] = None):
    """
    Returns the module associated with a given device(e.g., msadapter.device('cuda'), "mtia:0", "xpu", ...).
    If no device is given, return the module for the current accelerator or CPU if none is present.
    """
    if isinstance(device, msadapter.device):
        device_module_name = device.type
    elif isinstance(device, str):
        device_module_name = msadapter.device(device).type
    elif device is None:
        device_module_name = 'cuda'
    else:
        raise RuntimeError(f"Invalid value of device '{device}', expect msadapter.device, str, or None")
    if device_module_name != 'cuda':
        raise RuntimeError(f"Device '{device_module_name}' does not have a corresponding module registered as 'msadapter.{device_module_name}'.")
    return cuda

def is_warn_always_enabled(*args, **kwargs):
    return False

def set_warn_always(*args, **kwargs):
    pass

def is_autocast_cache_enabled():
    return True

def is_autocast_cpu_enabled():
    return False

def get_autocast_cpu_dtype():
    return msadapter.float32

def get_autocast_gpu_dtype():
    return msadapter.float32

def is_anomaly_enabled():
    return False

def is_anomaly_check_nan_enabled():
    return False

def set_anomaly_enabled(*args, **kwargs):
    pass

def npu_apply_fused_adamw_v2(param: Tensor,
                             grad: Tensor,
                             exp_avg: Tensor,
                             exp_avg_sq: Tensor,
                             max_exp_avg_sq: Tensor,
                             state_step: Tensor,
                             lr: float = 1e-3,
                             beta1: float = 0.9,
                             beta2: float = 0.999,
                             weight_decay: float = 0.0,
                             eps: float = 1e-8,
                             amsgrad: bool = False,
                             maximize: bool = False,
                             ):
    success = True
    state_step = state_step - 1
    adamw_opt = gen.AdamW()
    if max_exp_avg_sq is None:
        adamw_opt(param, exp_avg, exp_avg_sq, exp_avg_sq, grad, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)
    else:
        adamw_opt(param, exp_avg, exp_avg_sq, max_exp_avg_sq, grad, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)
    return success

def _fused_adamw_(
    params=None,
    grads=None,
    exp_avgs=None,
    exp_avg_sqs=None,
    max_exp_avg_sqs=None,
    state_steps=None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    amsgrad: bool,
    maximize: bool,
    grad_scale=None,
    found_inf=None
):
    npu_apply_fused_adamw_v2(
        params[0],
        grads[0],
        exp_avgs[0],
        exp_avg_sqs[0],
        max_exp_avg_sqs[0] if amsgrad else None,
        state_steps[0],
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
        maximize,
    )