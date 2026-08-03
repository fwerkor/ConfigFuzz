from typing import Any, Tuple, Union
import os
import mindspore
from mindspore import Tensor
from mindspore import default_generator, get_rng_state, set_rng_state, manual_seed
from mindspore.runtime import *
from mindspore.runtime import StreamCtx as stream

import msadapter
from msadapter.types import Device

from . import memory, config, tunable, random
from .memory import *
from .amp import *
from .config import *
from ..configs import support_bf16 as is_bf16_supported
from .typed_tensor import *


class DefaultGenerators:
    def __getitem__(self, idx):
        return default_generator

default_generators = DefaultGenerators()

def init():
    pass

def manual_seed_all(seed: int):
    manual_seed(seed)

def current_device():
    return 'Ascend'

def is_available():
    return True

def device_count():
    return mindspore.device_context.ascend.device_count()

def get_device_capability():
    return (8, 8)

device_map = {
    'cuda': 'Ascend',
    'npu': 'Ascend',
    'cpu': 'CPU'
}

def set_device(device):
    if os.getenv("USE_RAY", 'False').strip().lower() == 'true':
        if hasattr(device, "index"):
            mindspore.set_device("Ascend", device_id = device.index)
        else:
            mindspore.set_device("Ascend", device_id=device)

    def get_ms_device_type(device_type):
        ms_device_type = device_map.get(device_type, None)
        if ms_device_type is None:
            raise ValueError(f"Unsupported device type {device_type}, only 'cuda', 'npu' and 'cpu' is supported.")
        return ms_device_type

    if isinstance(device, str):
        splited_device = device.split(":")
        device_type = splited_device[0]
        device_id = int(splited_device[1]) if len(splited_device) > 1 else 0
        ms_device_type = get_ms_device_type(device_type)
        mindspore.set_device(device_target=ms_device_type, device_id=device_id)

    if not isinstance(device, str) and hasattr(device, "index") and device.index is not None:
        device_type = device.type if hasattr(device, "type") else None
        ms_device_type = get_ms_device_type(device_type)
        mindspore.set_device(device_target=ms_device_type, device_id=device.index)

def _get_device_target_and_id(device: Any, allow_cpu: bool = False) -> Tuple[str, int]:
    r"""Get the device target and id from :attr:`device`, which can be a msadapter.device object, a Python integer,
    or ``None``.
    """
    if device is None:
        return None, mindspore.context.get_context('device_id')

    if isinstance(device, int):
        return None, device

    if isinstance(device, str):
        device = msadapter.device(device)

    if not isinstance(device, msadapter.device):
        raise TypeError(f"Argument 'device' must be int, str or msadapter.device, not {type(device).__name__}")

    ms_device_type = device.type
    if ms_device_type not in device_map.values():
        ms_device_type = device_map.get(device.type, None)

    if allow_cpu:
        if ms_device_type not in ['Ascend', 'CPU']:
            raise ValueError(f"Expected a cuda, npu or cpu device, but got {device.type}")
    elif ms_device_type != 'Ascend':
        raise ValueError(f"Expected a cuda or npu device, but got {device.type}")

    if device.index is None:
        if mindspore.context.get_context('device_target') == ms_device_type:
            device.index = mindspore.context.get_context('device_id')
        else:
            device.index = 0

    return ms_device_type, device.index

if hasattr(mindspore._c_expression, 'AscendDeviceProperties'):
    mindspore._c_expression.AscendDeviceProperties.multi_processor_count = 1

def get_device_properties(device: Union[Device, str, int] = None) -> Any:
    device_target, device_id = _get_device_target_and_id(device)
    return mindspore.hal.get_device_properties(device_id, device_target)

def _lazy_call(callable, **kwargs):
    callable()

class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (msadapter.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = -1

    def __exit__(self, type: Any, value: Any, traceback: Any):
        return False

def synchronize():
    return mindspore.hal.synchronize()

def _try_initial_ascend():
    x = mindspore.tensor(1)
    _ = mindspore.ops.add(x, 0)


def memory_stats(device_target=None):
    res = mindspore.hal.memory_stats(device_target)
    if not res:
        _try_initial_ascend()
        res = mindspore.hal.memory_stats(device_target)

    stats = {
        "allocated_bytes.all.peak": res.get("max_allocated_memory", 0),
        "allocated_bytes.all.current": res.get("total_allocated_memory", 0),
        "num_alloc_retries": 0
    }
    return stats


def max_memory_allocated(device_target=None):
    res = mindspore.hal.memory_stats()
    if not res:
        _try_initial_ascend()
        res = mindspore.hal.memory_stats(device_target)

    return res.get("max_allocated_memory", 0)


def mem_get_info(device: Union[Device, int] = None) -> Tuple[int, int]:
    if not isinstance(device, int):
        device = mindspore.context.get_context("device_id")

    res = mindspore.hal.get_device_properties(device)
    if res.total_memory == 0:
        _try_initial_ascend()
        res = mindspore.hal.get_device_properties(device)

    return (res.free_memory, res.total_memory)


def reset_peak_memory_stats(device_target=None):
    mindspore.runtime.reset_peak_memory_stats()

def is_initialized():
    return True

def set_stream(stream):
     mindspore.runtime.set_cur_stream(stream)

def _is_compiled():
    return False

def get_device_name(device=None):
    return "Ascend"

def set_compile_mode(jit_compile: bool):
    pass

def _lazy_init():
    return True

def _sleep(cycles):
    pass