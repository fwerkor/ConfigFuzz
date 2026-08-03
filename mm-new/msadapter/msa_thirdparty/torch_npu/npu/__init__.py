import warnings
import traceback
import mindspore
from mindspore.runtime import *
from mindspore.runtime import StreamCtx as stream
from .utils import device
from .utils import current_stream
from .mstx import mstx
from .streams import *
from .npu_config import *

from .random import manual_seed_all
from torch.random import set_rng_state, get_rng_state, manual_seed, initial_seed
from torch._C.typed_tensor import *


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
    }
    return stats


def current_device():
    pass 

def _lazy_call(cb):
    cb()

def npu_fusion_attention(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                                atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647,
                                inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
    pass

def empty_cache():
    return mindspore.hal.empty_cache()

def get_device_name(device:int = None) -> str:
    # mindspore.hal.get_device_name only support 'device' to be 'int'
    return mindspore.hal.get_device_name(device_id=device)

def init():
    warnings.warn("The 'init()' function dose NOT perform actual initialization.")
    _lazy_init()

def _lazy_init():
    pass
