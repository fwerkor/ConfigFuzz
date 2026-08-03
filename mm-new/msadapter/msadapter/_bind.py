from mindspore import float16, bfloat16
from .configs import DEFAULT_DTYPE
import mindspore
import numpy as np
import msadapter

AUTO_CAST_DTYE = {
    'cuda': float16,
    'cpu': bfloat16,
    'npu': float16
}

def set_autocast_dtype(device_type, dtype):
    assert device_type in AUTO_CAST_DTYE.keys(), f'{device_type} is not in {AUTO_CAST_DTYE.keys()}'
    AUTO_CAST_DTYE[device_type] = dtype

def get_autocast_dtype(device_type):
    return AUTO_CAST_DTYE[device_type]

def set_default_dtype(dtype):
    """set default dtype"""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype

def get_default_dtype():
    """get default dtype"""
    return DEFAULT_DTYPE


class finfo:
    def __init__(self, bits, min, max, eps, tiny, smallest_normal, resolution, dtype):
        self.bits = bits
        self.min = min
        self.max = max
        self.eps = eps
        self.tiny = tiny
        self.smallest_normal = smallest_normal
        self.resolution = resolution
        self.dtype = dtype

    def __repr__(self):
        return f'finfo(resolution={self.resolution}, min={self.min}, max={self.max}, eps={self.eps}, smallest_normal={self.smallest_normal}, tiny={self.tiny}, dtype={self.dtype})'


_float16_np_finfo = np.finfo(np.float16)
_float32_np_finfo = np.finfo(np.float32)
_float64_np_finfo = np.finfo(np.float64)

_float8_info = finfo(bits=8, resolution=1, min=-448, max=448, eps=0.125,
                     smallest_normal=0.015625, tiny=0.015625, dtype="float8_e4m3fn")

_bfloat16_info = finfo(bits=16, resolution=0.01, min=-3.3895313892515355e+38, max=3.3895313892515355e+38, eps=0.0078125,
                       smallest_normal=1.1754943508222875e-38, tiny=1.1754943508222875e-38, dtype="bfloat16")
_float16_info = finfo(bits=16, resolution=_float16_np_finfo.resolution.item(), min=_float16_np_finfo.min.item(),
                      max=_float16_np_finfo.max.item(), eps=0.0009765625,
                      smallest_normal=6.103515625e-05, tiny=6.103515625e-05, dtype="float16")
_float32_info = finfo(bits=32, resolution=_float32_np_finfo.resolution.item(), min=_float32_np_finfo.min.item(),
                      max=_float32_np_finfo.max.item(),
                      eps=1.1920928955078125e-07, smallest_normal=1.1754943508222875e-38, tiny=1.1754943508222875e-38,
                      dtype="float32")
_float64_info = finfo(bits=64, resolution=_float64_np_finfo.resolution.item(), min=_float64_np_finfo.min.item(),
                      max=_float64_np_finfo.max.item(),
                      eps=2.220446049250313e-16, smallest_normal=2.2250738585072014e-308, tiny=2.2250738585072014e-308,
                      dtype="float64")
_finfo_dtype = {
    mindspore.float8_e4m3fn: _float8_info,
    mindspore.bfloat16: _bfloat16_info,
    mindspore.float16: _float16_info,
    mindspore.float32: _float32_info,
    mindspore.float64: _float64_info,
    mindspore.complex64: _float32_info,
    mindspore.complex128: _float64_info,
    float: _float64_info,
}


def finfo(dtype=mindspore.float32):
    if dtype not in _finfo_dtype.keys():
        raise TypeError(
            f"finfo(): argument 'dtype' (position 1) must be floating-point or complex type, but got {dtype}")
    return _finfo_dtype[dtype]


def iinfo(dtype):
    if dtype == int:
        return np.iinfo(int)
    return np.iinfo(msadapter._utils.dtype_to_nptype(dtype))
