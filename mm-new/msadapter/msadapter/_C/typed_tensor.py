from mindspore.common.tensor import _TensorMeta
from .. import Tensor
from .. import float32, float64, float16, bfloat16, uint8, int8, int16, int32, int64, bool_


class _TypedTensorMeta(_TensorMeta):
    def __instancecheck__(self, instance):
        if not isinstance(instance, Tensor):
            return False
        if not instance._ms_device.split(':')[0] == self._device_type:
            return False
        return instance.dtype == self.dtype

class _TypedTensorBase(Tensor, metaclass=_TypedTensorMeta):
    _device_type = 'CPU'
    def __new__(cls, data, device=None):
        return Tensor(data, dtype=cls.dtype)

FloatTensor = type('FloatTensor', (_TypedTensorBase,), {'dtype': float32})
DoubleTensor = type('DoubleTensor', (_TypedTensorBase,), {'dtype': float64})
HalfTensor = type('HalfTensor', (_TypedTensorBase,), {'dtype': float16})
BFloat16Tensor = type('BFloat16Tensor', (_TypedTensorBase,), {'dtype': bfloat16})
ByteTensor = type('ByteTensor', (_TypedTensorBase,), {'dtype': uint8})
CharTensor = type('CharTensor', (_TypedTensorBase,), {'dtype': int8})
ShortTensor = type('ShortTensor', (_TypedTensorBase,), {'dtype': int16})
IntTensor = type('IntTensor', (_TypedTensorBase,), {'dtype': int32})
LongTensor = type('LongTensor', (_TypedTensorBase,), {'dtype': int64})
BoolTensor = type('BoolTensor', (_TypedTensorBase,), {'dtype': bool_})
