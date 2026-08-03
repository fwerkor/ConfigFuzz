"""creation ops"""
import msadapter
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
import mindspore
try:
    from mindspore._c_expression import Tensor as CTensor # pylint: disable=no-name-in-module, import-error
except:
    from mindspore._c_expression import TensorPy as CTensor # pylint: disable=no-name-in-module, import-error
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from ..configs import use_pyboost, on_orange_pi, MS26, MS271, HAS_OP_PLUGIN
from .._bind import get_default_dtype
from ._inner import get_item


_TypeDict = {
    mindspore.float16: np.float16,
    mindspore.float32: np.float32,
    mindspore.float64: np.float64,
    mindspore.bfloat16: np_bfloat16,
    mindspore.int8: np.int8,
    mindspore.int16: np.int16,
    mindspore.int32: np.int32,
    mindspore.int64: np.int64,
    mindspore.uint8: np.uint8,
    mindspore.bool_: np.bool_,
    mindspore.complex64: np.complex64,
    mindspore.complex128: np.complex128,
}

py2dtype = {
    bool: mindspore.bool_,
    int: mindspore.int64,
    float: mindspore.float64
}

def as_strided(self, size, stride, storage_offset=None):
    if len(size) != len(stride):
        raise RuntimeError("mismatch in length of strides and shape.")
    index = np.arange(0, size[0] * stride[0], stride[0])
    for i in np.arange(1, len(size)):
        tmp = np.arange(0, size[i] * stride[i], stride[i])
        index = np.expand_dims(index, -1)
        index = index + tmp
    if storage_offset is not None:
        index = index + storage_offset

    if index.size == 0:
        input_indices = mindspore.numpy.empty(index.shape, dtype=mindspore.int32)
    else:
        input_indices = mindspore.tensor(index.astype(np.int32))
    out = ops.gather(self.reshape(-1), input_indices, 0)
    return out

# from_numpy
def from_numpy(ndarray):
    return mindspore.Tensor(ndarray)

# frombuffer

# zeros
_zeros = ops.Zeros()
has_zeros = hasattr(mindspore.mint, 'zeros')
def zeros(*size, dtype=None, device=None, requires_grad=False, **kwargs):
    if dtype is None:
        dtype = get_default_dtype()
    if isinstance(dtype, type):
        dtype = py2dtype[dtype]
    if len(size) == 0:
        size = kwargs.get('size', None)
        if size == () or size == []:
            size = ((),)
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    if use_pyboost() and has_zeros:
        if device == 'cpu':
            tensor = mindspore.Tensor(np.zeros(size), dtype=dtype)
        else:
            tensor = mindspore.mint.zeros(size, dtype=dtype)
    else:
        size = tuple(size)
        tensor = _zeros(size, dtype)
    tensor.requires_grad = requires_grad
    return tensor


# zeros_like
has_zeros_like = hasattr(mindspore.mint, 'zeros_like')
def zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=None, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if (HAS_OP_PLUGIN or use_pyboost()) and has_zeros_like:
        tensor = mindspore.mint.zeros_like(input, dtype=dtype)
    else:
        tensor = ops.zeros_like(input, dtype=dtype)
    if requires_grad:
        tensor.requires_grad = requires_grad
    return tensor


# ones
_ones = ops.Ones()
has_ones = hasattr(mindspore.mint, 'ones')
def ones(*size, dtype=None, device=None, requires_grad=False):
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    if dtype is None:
        dtype = get_default_dtype()
    if dtype == bool:
        dtype = mindspore.bool_
    if use_pyboost() and has_ones:
        tensor = mindspore.mint.ones(size, dtype=dtype)
    else:
        tensor = _ones(size, dtype)
    tensor.requires_grad = requires_grad
    return tensor


# ones_like
has_ones_like = hasattr(mindspore.mint, 'ones_like')
def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=None, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if use_pyboost() and has_ones_like:
        tensor = mindspore.mint.ones_like(input, dtype=dtype)
    else:
        tensor = ops.ones_like(input, dtype=dtype)
    if requires_grad:
        tensor.requires_grad = requires_grad
    return tensor

# arange
has_arange = hasattr(mindspore.mint, 'arange')
def arange(start=0, end=None, step=1, *, out=None, dtype=None, layout=msadapter.strided, device=None, requires_grad=False):
    if on_orange_pi() and dtype in (None, mindspore.int64):
        dtype = mindspore.int32
    start = get_item(start)
    end = get_item(end)
    step = get_item(step)
    if use_pyboost() and has_arange:
        tensor = mindspore.mint.arange(start, end, step, dtype=dtype)
    else:
        tensor = ops.arange(start, end, step, dtype=dtype)
    tensor.requires_grad = requires_grad
    return tensor

# range
def range(start=0, end=None, step=1, *, out=None, dtype=None, layout=msadapter.strided, device=None, requires_grad=False):
    if end is None:
        start, end = 0, start
    out = ops.range(start, end + 1, step)
    if dtype is not None:
        out = out.to(dtype)
    out.requires_grad = requires_grad
    return out

# linspace
has_linspace = hasattr(mindspore.mint, 'linspace')
def linspace(start, end, steps, *, out=None, dtype=None, layout=msadapter.strided, device=None, requires_grad=False):
    if dtype is None:
        dtype = mindspore.float32
    if use_pyboost() and has_linspace:
        if isinstance(steps, mindspore.Tensor):
            steps = steps.item()
        tensor = mindspore.mint.linspace(start, end, steps, dtype=dtype)
    else:
        tensor = ops.linspace(start, end, steps).to(dtype)
    tensor.requires_grad = requires_grad
    return tensor

# logspace
def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=msadapter.strided, device=None, requires_grad=False):
    tensor = ops.logspace(start, end, steps, base, dtype=dtype)
    tensor.requires_grad = requires_grad
    return tensor

# eye
has_eye = hasattr(mindspore.mint, 'eye')
def eye(n, m=None, *, out=None, dtype=None, layout=msadapter.strided, device=None, requires_grad=False):
    if use_pyboost() and has_eye:
        tensor = mindspore.mint.eye(n, m, dtype)
    else:
        tensor = ops.eye(n, m, dtype)
    tensor.requires_grad = requires_grad
    return tensor

# empty
has_empty = hasattr(mindspore.mint, 'empty') and MS26
def empty(*size, dtype=None, device=None, requires_grad=False, pin_memory=False, **kwargs):
    size = kwargs.get('size', size)
    if device is None:
        device = "CPU"
    if isinstance(size[0], (tuple, list)):
        size = size[0]

    if not size:
        pass
    else:
        if isinstance(size[0], mindspore.Tensor):
            if len(size) == 1:
                size = size[0].asnumpy().tolist()
            else:
                size = [sz.item() for sz in size]

    if isinstance(size, (tuple, list)) and len(size) > 0:
        size = tuple(data.item() if isinstance(data, mindspore.Tensor) else data for data in size)
    if dtype is None:
        dtype = get_default_dtype()

    if device:
        if not isinstance(device, str) and hasattr(device, "type"):
            device = device.type
        if device.lower() == 'cpu':
            device = device.upper()
        elif device.lower() in ['cuda', 'npu']:
            device = 'Ascend'

    # To avoid the problem in irecv and recv of using empty.

    if has_empty:
        out = mindspore.mint.empty(size, dtype=dtype, device=device, pin_memory=pin_memory)
    else:
        out = CTensor(dtype, size)
        out = mindspore.Tensor(out)
    out.requires_grad = requires_grad
    return out

# empty_like
has_empty_like = hasattr(mindspore.mint, 'empty_like')
def empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=None, memory_format=None, pin_memory=False):
    if MS271:
        tensor = mindspore.mint.empty_like(input, dtype=dtype, device=device, pin_memory=pin_memory)
    else:
        tensor = mindspore.mint.empty_like(input, dtype=dtype, device=device)
    if requires_grad:
        tensor.requires_grad = requires_grad
    return tensor

# empty_strided

# full
has_full = hasattr(mindspore.mint, 'full')
def full(size, fill_value, *, out=None, dtype=None, layout=msadapter.strided, device=None, requires_grad=False):
    if isinstance(fill_value, np.generic):
        fill_value = fill_value.item()
    if use_pyboost() and has_full:
        tensor = mindspore.mint.full(size, fill_value, dtype=dtype)
    else:
        tensor = ops.full(size, fill_value, dtype=dtype)
    tensor.requires_grad = requires_grad
    return tensor

# full_like
has_full_like = hasattr(mindspore.mint, 'full_like')
def full_like(input, fill_value, *, dtype=None, layout=msadapter.strided, device=None, requires_grad=None, memory_format=msadapter.preserve_format):
    if use_pyboost() and has_full_like:
        tensor = mindspore.mint.full_like(input, fill_value, dtype=dtype)
    else:
        if dtype is None:
            dtype = input.dtype
        tensor = full(input.shape, fill_value, dtype=dtype)
    if requires_grad:
        tensor.requires_grad = requires_grad
    return tensor

# quantize_per_tensor


# quantize_per_channel


# dequantize


# complex
def complex(real, imag):
    _complex = _get_cache_prim(ops.Complex)()
    return _complex(real, imag)

# polar
has_polar = hasattr(mindspore.mint, 'polar')
def polar(abs, angle):
    if use_pyboost() and has_polar:
        if abs.dtype == mindspore.float64 or angle.dtype == mindspore.float64:
            return mindspore.mint.polar(abs.to(mindspore.float32), angle.to(mindspore.float32)).to(mindspore.float64)
        return mindspore.mint.polar(abs, angle)
    return ops.polar(abs, angle)

# heaviside
def heaviside(input, values):
    return ops.heaviside(input, values)


def frombuffer(buffer, *, dtype=None, count=-1, offset=0, requires_grad=False):
    tensor = mindspore.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
    tensor.requires_grad = requires_grad
    return tensor


__all__ = ['arange', 'as_strided', 'complex', 'empty', 'empty_like',
           'eye', 'from_numpy', 'full', 'full_like', 'frombuffer',
           'heaviside', 'linspace', 'logspace', 'ones', 'ones_like',
           'polar', 'range', 'zeros', 'zeros_like'
           ]