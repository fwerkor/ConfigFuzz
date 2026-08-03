import msadapter
from msadapter import Tensor
from msadapter import float32, float64, float16, bfloat16, uint8, int8, int16, int32, int64, bool_
from msadapter._C import typed_tensor


class _CudaTypedTensorBase(Tensor, metaclass=typed_tensor._TypedTensorMeta):
    _device_type = 'Ascend'
    def __new__(cls, data, device=None):
        tensor = Tensor(data, dtype=cls.dtype)
        return msadapter._tensor._ms_to(tensor, 'Ascend')

FloatTensor = type('FloatTensor', (_CudaTypedTensorBase,), {'dtype': float32})
DoubleTensor = type('DoubleTensor', (_CudaTypedTensorBase,), {'dtype': float64})
HalfTensor = type('HalfTensor', (_CudaTypedTensorBase,), {'dtype': float16})
BFloat16Tensor = type('BFloat16Tensor', (_CudaTypedTensorBase,), {'dtype': bfloat16})
ByteTensor = type('ByteTensor', (_CudaTypedTensorBase,), {'dtype': uint8})
CharTensor = type('CharTensor', (_CudaTypedTensorBase,), {'dtype': int8})
ShortTensor = type('ShortTensor', (_CudaTypedTensorBase,), {'dtype': int16})
IntTensor = type('IntTensor', (_CudaTypedTensorBase,), {'dtype': int32})
LongTensor = type('LongTensor', (_CudaTypedTensorBase,), {'dtype': int64})
BoolTensor = type('BoolTensor', (_CudaTypedTensorBase,), {'dtype': bool_})
