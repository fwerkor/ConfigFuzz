import math
import warnings
from types import FunctionType, MethodType
from typing import Sequence, List
import numpy as np

import msadapter
import mindspore
import mindspore.ops.function.nn_func as nn_func
from mindspore._c_expression import typing
from mindspore._c_expression.typing import Type
from mindspore import Tensor, ops, mint
from mindspore._c_expression import pyboost_detach
from mindspore.common.hook_handle import _TensorHookHandle
from mindspore.common._register_for_tensor import tensor_operator_registry

try:
    from mindspore.common._stub_tensor import StubTensor
except:
    class StubTensor: pass

try:
    from mindspore._c_expression import TensorPy as Tensor_
except:
    from mindspore._c_expression import Tensor as Tensor_
from .configs import use_pyboost, MS271, MS28

from ._utils import _rebuild_tensor_v2
from ._C.size import Size
from .ops import (transpose, mean, repeat_interleave, unsqueeze, pow, clamp,
                  split, norm, reshape, softmax, ceil, cos, log, round,
                  sin, sqrt, square, sub, trunc, abs, exp, floor, gather,
                  matmul, mul, ne, absolute, amax, ones, equal, nan_to_num, addcdiv,
                  addcmul, lerp, triu, as_strided, unflatten as _unflatten)
from .ops._inner import get_item
from .types import device as adapter_device


MS_PT_DTYPE_MAP = {
    'Float32': 'torch.cuda.FloatTensor',
    'BFloat16': 'torch.cuda.BFloat16Tensor',
    'Float16': 'torch.cuda.HalfTensor',
}

DTYPE_ELEMENT_SIZE_MAP = {
    mindspore.float64: 8,
    mindspore.int64: 8,
    mindspore.int32: 4,
    mindspore.float32: 4,
    mindspore.int16: 2,
    mindspore.bfloat16: 2,
    mindspore.float16: 2,
    mindspore.bool_: 1,
    mindspore.int8: 1,
    mindspore.uint8: 1
}

PT_MS_DEVICE_TYPE_MAP = {
    'cuda': 'Ascend',
    'npu': 'Ascend',
    'Ascend': 'Ascend',
    'cpu': 'CPU'
}

def register_to_tensor(func=None, *, func_name=None, func_names: List[str] = None):
    def decorator(f):
        nonlocal func_name, func_names
        if func_name is not None and func_names is not None:
            raise RuntimeError("argument 'func_name' and 'func_names' are mutually exclusive!")
        if func_names and isinstance(func_names, list):
            for func_name in func_names:
                setattr(Tensor, func_name, f)
                setattr(StubTensor, func_name, f)
        else:
            if func_name is None:
                func_name = f.fget.__name__ if isinstance(f, property) else f.__name__
            setattr(Tensor, func_name, f)
            setattr(StubTensor, func_name, f)
        return f
    if func is None:
        return decorator
    else:
        return decorator(func)

@register_to_tensor
def _typed_storage(self):
    return TypedStorageProxy(self)

class TypedStorageProxy:
    def __init__(self, tensor):
        self.tensor = tensor 
        self.dtyped_size = msadapter._utils._element_size(self.tensor.dtype)
    
    def _size(self):
        return self.tensor.untyped_storage().size() // self.dtyped_size

    def _resize_(self, new_size):
        return self.tensor.untyped_storage().resize_(new_size * self.dtyped_size)
    
@register_to_tensor
def unflatten(self, dim, size):
    return _unflatten(self, dim, size)

@register_to_tensor
def transpose_(self, dim0, dim1):
    return self.assign_value(transpose(self, dim0, dim1))

@register_to_tensor
def triu_(self, diagonal=0):
    return self.copy_(triu(self, diagonal=diagonal))

@register_to_tensor
def resize_(self, *sizes, memory_format=None):
    return self.copy_(self.resize(sizes))

@register_to_tensor
def masked_scatter_(self, mask, tensor):
    return self.copy_(self.masked_scatter(mask, tensor))

Tensor.as_strided = as_strided
StubTensor.as_strided = as_strided


@register_to_tensor
def nan_to_num_(self, nan=0.0, posinf=None, neginf=None):
    return self.copy_(nan_to_num(self, nan, posinf, neginf))


@register_to_tensor
def lerp_(self, end, weight):
    return self.copy_(lerp(self, end, weight))


@register_to_tensor
def addcdiv_(self, tensor1, tensor2, *, value=1):
    return self.copy_(addcdiv(self, tensor1, tensor2, value=value))


@register_to_tensor
def addcmul_(self, tensor1, tensor2, *, value=1):
    return self.copy_(addcmul(self, tensor1, tensor2, value=value))


@register_to_tensor
def abs_(self):
    return self.copy_(abs(self))


@register_to_tensor(func_name='bool')
def _bool(self):
    return self.to(mindspore.bool_).detach()

@register_to_tensor
def register_hook(self, hook):
    def check_hook_fn(hook_fn):
        if not (isinstance(hook_fn, (FunctionType, MethodType)) or callable(hook_fn)):
            raise TypeError(f"When using 'hook_type(hook_fn)', the type of 'hook_fn' must be python "
                            f"function or callable object, but got {type(hook_fn)}.")
        if hasattr(hook_fn, "__code__") and hook_fn.__code__.co_name == "staging_specialize":
            raise TypeError(f"Decorating hook function {hook_fn.__name__} with '@jit' is not supported.")
    check_hook_fn(hook)
    handle = _TensorHookHandle(self)
    handle.id = Tensor_.register_hook(self, hook)
    return handle

@register_to_tensor
def __truediv__(self, other, *, rounding_mode=None, out=None):
    if mindspore.get_context('device_target') == 'CPU':
        self = self.numpy()
        return Tensor.from_numpy(np.divide(self, other, out=None))
    if use_pyboost() and hasattr(mindspore.mint, 'div'):
        return mindspore.mint.div(self, other, rounding_mode=rounding_mode)
    return mindspore.ops.div(self, other, rounding_mode=rounding_mode)

@register_to_tensor
def __rtruediv__(self, other, *, rounding_mode=None, out=None):
    if use_pyboost() and hasattr(mindspore.mint, 'reciprocal'):
        return mindspore.mint.reciprocal(self) * other
    return mindspore.ops.reciprocal(self) * other

Tensor.__rdiv__ = __rtruediv__

@register_to_tensor
def imag(self):
    return tensor_operator_registry.get('imag')(self)

@register_to_tensor
def real(self):
    return tensor_operator_registry.get('real')(self)

@register_to_tensor
def to_sparse(self):
    return self

@register_to_tensor
@property
def volatile(self):
    warnings.warn("volatile was removed(Variable.volatile is always False)", UserWarning)
    return False

@register_to_tensor
@property
def is_nested(self):
    if not hasattr(self, '_is_nested'):
        self._is_nested = False
    return self._is_nested

@register_to_tensor
@property
def is_quantized(self):
    if not hasattr(self, '_is_quantized'):
        self._is_quantized = False
    return self._is_quantized

@register_to_tensor
@property
def is_mps(self):
    if not hasattr(self, '_is_mps'):
        self._is_mps = False
    return self._is_mps

@register_to_tensor
@property
def requires_grad(self):
    return self._requires_grad

@register_to_tensor
@requires_grad.setter
def requires_grad(self, value=True):
    if not isinstance(value, bool):
        raise TypeError("The argument `requires_grad` must be bool type")
    self._requires_grad = value

@register_to_tensor
@property
def grad(self):
    if not self.is_leaf and self.requires_grad:
        warnings.warn(
            "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. "
            "Its .grad attribute won't be populated during autograd.backward(). "
            "If you indeed want the .grad field to be populated for a non-leaf Tensor, "
            "use .retain_grad() on the non-leaf Tensor."
        )
    return self._grad

@register_to_tensor
@grad.setter
def grad(self, value):
    self._grad = value

@register_to_tensor
def retain_grad(self):
    return self._retain_grad()

@register_to_tensor
@property
def is_leaf(self):
    return self._is_leaf

@register_to_tensor
def requires_grad_(self, requires_grad=True):
    if not hasattr(self, "requires_grad"):
        raise AttributeError("Input must have 'requires_grad' attribute.")
    if requires_grad == self.requires_grad:
        return self
    if not self.is_leaf:
        raise RuntimeError("you can only change requires_grad flags of leaf variables. "
                           "If you want to use a computed variable in a subgraph that doesn't require differentiation "
                           "use var_no_grad = var.detach().")
    self.requires_grad = requires_grad
    return self

@register_to_tensor
@property
def retains_grad(self):
    return self._retains_grad

@register_to_tensor
@property
def grad_fn(self):
    if self._grad_node and self._grad_node.is_leaf():
        return None
    return self._grad_node

@register_to_tensor
@property
def output_nr(self):
    return self._output_index

@register_to_tensor
def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    msadapter.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)

@register_to_tensor
@property
def is_floating_point(self):
    return isinstance(self.dtype, (typing.Float, typing.BFloat))

@register_to_tensor
def is_floating_point(self):
    return isinstance(self.dtype, (typing.Float, typing.BFloat))

@register_to_tensor(func_name='squeeze')
def squeeze_(self, dim=None):
    if dim is None:
        dim = ()
    tensor = mindspore.mint.squeeze(self, dim)
    tensor._base = self
    return tensor

@register_to_tensor
def new(self, data=None):
    if data is None:
        return Tensor([], dtype=self.dtype)
    return Tensor(data, dtype=self.dtype)

Tensor.new = new
StubTensor.new = new

npu_fill_ = Tensor.fill_

CACHE_DEVICE_TARGET = None
def fill_(self, data=None):
    if data is None:
        raise ValueError("fill_(): data cannot be None.")
    global CACHE_DEVICE_TARGET
    if CACHE_DEVICE_TARGET is None or CACHE_DEVICE_TARGET == "CPU":
        CACHE_DEVICE_TARGET = mindspore.get_context('device_target')
    if CACHE_DEVICE_TARGET == 'CPU':
        np_data = np.ones(self.shape, dtype=np.float32) * data
        tensor_data = mindspore.Tensor(np_data, dtype=self.dtype)
        self.copy_(tensor_data)
    else:
        npu_fill_(self, data)
    return self

Tensor.fill_ = fill_
StubTensor.fill_ = fill_

def is_shared(*arg, **kwarg):
    return False

@register_to_tensor(func_name='type')
def type_(self, dtype=None):
    if dtype is None:
        return MS_PT_DTYPE_MAP[str(self.dtype)]
    return self.to(dtype)

@register_to_tensor
@property
def shape(self):
    return Size(self._shape, False)

@register_to_tensor
def to_dense(self):
    return self


@register_to_tensor
def numel(self):
    return math.prod(self.shape)

@register_to_tensor
def nelement(self):
    return math.prod(self.shape)

@register_to_tensor(func_names=["cuda", "npu"])
def move_to_cuda(self, non_blocking=False):
    return Tensor(self.move_to("Ascend", not non_blocking), requires_grad=self.requires_grad)

@register_to_tensor(func_name="cpu")
def move_to_cpu(self, non_blocking=False):
    return self.contiguous().move_to("CPU")

@register_to_tensor
def size(self, dim=None):
    if dim is None:
        return self.shape
    if not isinstance(dim, int):
        raise TypeError(f'`dim` must be int but got {type(dim)}')
    return self._shape[dim]

@register_to_tensor
def dim(self):
    return self.ndim

@register_to_tensor
def clone(self):
    return self.copy()

@register_to_tensor
def log_softmax(self, dim=-1, dtype=None):
    if use_pyboost():
        return mindspore.mint.nn.functional.log_softmax(self, dim=dim, dtype=dtype)
    out = ops.log_softmax(self, dim)
    if dtype is not None:
        out = out.to(dtype)
    return out

@register_to_tensor
def narrow(self, dim, start, length):
    if not isinstance(start, int):
        start = start.item()
    if not isinstance(length, int):
        length = length.item()
    if use_pyboost() and hasattr(mindspore.mint, 'narrow'):
        tensor = mindspore.mint.narrow(self, dim, start, length)
    else:
        tensor = ops.narrow(self, dim, start, length)
    tensor._base = self
    return tensor

@register_to_tensor
def __deepcopy__(self, memo):
    if not self.is_leaf:
        raise RuntimeError(
            "Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the momoent."
            "If you were attempting to deepcopy a module, this may be because of a torch.nn.utils.weight_norm usage, "
            "see https://github.com/pytorch/pytorch/pull/103001.")
    new_tensor = self.clone().detach()
    new_tensor.requires_grad = self.requires_grad
    memo[id(self)] = new_tensor
    return new_tensor

@register_to_tensor
def __rmul__(self, other):
    if isinstance(other, (str, list)):
        return other * self._item()
    return self.__mul__(other)

@register_to_tensor
def __mul__(self, other):
    return mul(self, other)

@register_to_tensor
def __iter__(self):
    for i in range(len(self)):
        yield self[i]

if not hasattr(Tensor, 'div_'):
    @register_to_tensor(func_name="div_")
    def mock_div_(self, value, *, rounding_mode=None):
        out = self.div(value, rounding_mode=rounding_mode)
        self.assign_value(out)

@register_to_tensor
def __reduce_ex__(self, protocol):
    tensor = self
    if not self.is_contiguous():
        tensor = self.contiguous()

    if isinstance(tensor, StubTensor):
        data = Tensor_(tensor.stub_sync())
    else:
        data = Tensor_(tensor)
    storage_offset = 0
    size = data._shape
    stride = data.stride()
    requires_grad = False
    args = (data, storage_offset, size, stride, requires_grad, None, None)
    return (
        _rebuild_from_type_v2, (_rebuild_tensor_v2, type(tensor), args, None))

def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    return ret

@register_to_tensor
def detach(self):
    return pyboost_detach(self)

@register_to_tensor
def detach_(self):
    return mindspore.ops.auto_generate.inplace_stop_gradient(self)

@register_to_tensor
def values(self):
    warnings.warn("MindSpore's values() and _values() is not supported now. This func will return the tensor itself directly!")
    return self

@register_to_tensor
def _values(self):
    warnings.warn("MindSpore's values() and _values() is not supported now. This func will return the tensor itself directly!")
    return self

@register_to_tensor
@property
def T(self):
    rank_list = [i for i in range(self.ndim - 1, -1, -1)]
    return self.permute(rank_list)

@register_to_tensor
def __pow__(self, exponent):
    return pow(self, exponent)

@register_to_tensor(func_name='float')
def _float(self):
    if self.dtype == mindspore.float32:
        return self
    return self.to(mindspore.float32)

@register_to_tensor(func_name='norm')
def norm_(self, p='fro', dim=None, keepdim=False, dtype=None):
    return norm(self, p=p, dim=dim, keepdim=keepdim, dtype=dtype)

@register_to_tensor(func_name='record_stream')
def _record_stream(tensor, ctx_stream):
    return -1

@register_to_tensor
def data_ptr(self):
    return self._data_ptr()

@register_to_tensor
def __imul__(self, other):
    self.mul_(other)
    return self

@register_to_tensor
def element_size(self,):
    return DTYPE_ELEMENT_SIZE_MAP[self.dtype]

if not hasattr(Tensor, 'exponential_'):
    @register_to_tensor(func_name='exponential_')
    def _exponential(self, lambd=1.0, *, generator=None):
        if generator is not None:
            raise ValueError("`generator` can not be supported.")
        output = np.random.exponential(scale=lambd, size=self.shape)
        self.copy_(mindspore.Tensor(output).to(self.dtype))

@register_to_tensor(func_name='is_shared')
def is_shared_(self,):
    return False

def _device_to_ms(device):
    if isinstance(device, str):
        torch_device_type = msadapter.device(device).type
        return PT_MS_DEVICE_TYPE_MAP[torch_device_type]
    elif isinstance(device, msadapter.device):
        if device.type == "meta":
            return None
        return device.type
    else:
        raise TypeError(f"Argument 'device' must be str or torch.device, not {type(device).__name__}")

Tensor._ms_device = Tensor.device

@register_to_tensor
@property
def device(self):
    msadapter_device = adapter_device(self._ms_device)
    if msadapter_device.type == 'Ascend':
        msadapter_device.type = 'npu'
    if msadapter_device.type == 'CPU':
        msadapter_device.type = 'cpu'
    return msadapter_device

if MS271:
    _ms_to = mindspore.Tensor.to
    @register_to_tensor(func_name="to")
    def _to_with_device(self, *args, **kwargs):
        if len(args) > 0:
            arg0 = args[0]
            if isinstance(arg0, (int, str, msadapter.device)):
                args = (_device_to_ms(arg0), ) + args[1:]

        device = kwargs.get("device", None)
        if device is not None:
            kwargs['device'] = _device_to_ms(device)

        return _ms_to(self, *args, **kwargs)
else:
    @register_to_tensor(func_name="to")
    def to_(self, *args, **kwargs):
        dtype_to = None
        if len(args) > 0:
            if isinstance(args[0], Type):
                dtype_to = args[0]
            elif isinstance(args[0], Tensor):
                dtype_to = args[0].dtype
            elif len(args) > 1 and isinstance(args[1], Type):
                dtype_to = args[1]

        if dtype_to is None:
            dtype_to = kwargs.get("dtype", None)
        if dtype_to is None:
            other = kwargs.get("other", None)
            dtype_to = other.dtype if other is not None else None
        if dtype_to is not None:
            return mindspore.ops.cast(self, dtype_to)
        return self

@register_to_tensor
def get_device(self):
    return mindspore.get_current_device().device_id

class ReturnTypes:
    def __init__(self, inputs):
        self.values = inputs[0]
        self.indices = inputs[1]

    def __iter__(self):
        yield self.values
        yield self.indices

    def __getitem__(self, index):
        if index == 0:
            return self.values
        elif index == 1:
            return self.indices
        else:
            raise IndexError("index out of range")


@register_to_tensor(func_name='max')
def max_(self, *args, **kwargs):
    res = mindspore.mint.max(self, *args, **kwargs)
    if isinstance(res, tuple):
        return ReturnTypes(res)
    return res

@register_to_tensor(func_name='is_conj')
def is_conj_(self):
    warnings.warn("MindSpore's conj() and is_conj() is different from PyTorch internally, current conj() and is_conj() are not supported")
    return False

@register_to_tensor()
def new_ones(self, *size, dtype=None, device=None, requires_grad=False, layout=None, pin_memory=False):
    if dtype is None:
        dtype = self.dtype
    if device is None:
        device = self.device
    if isinstance(size[0], tuple):
        size = size[0]
    return msadapter.ones(*size, dtype=dtype, device=device, requires_grad=requires_grad)

@register_to_tensor()
def new_zeros(self, *size, dtype=None, device=None, requires_grad=False, layout=msadapter.strided, pin_memory=False):
    if dtype is None:
        dtype = self.dtype
    if device is None:
        device = self.device
    if isinstance(size[0], tuple):
        size = size[0]
    return msadapter.zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad)

@register_to_tensor
@property
def layout(self):
    warnings.warn("MSAdapter does not support msadapter.layout currently, tensor.layout will return strided directly!")
    return msadapter.strided

@register_to_tensor
def relu_(self):
    return nn_func.relu_(self)

@register_to_tensor
def relu(self):
    return nn_func.relu(self)


@register_to_tensor
def split_with_sizes(self, sizes, dim=0):
    return split(self, sizes, dim)

@register_to_tensor
@property
def is_npu(self):
    warnings.warn("MSAdapter does not support Tensor.is_npu currently, Tensor.is_npu will return True.")
    return True

@register_to_tensor
def masked_fill(self, mask, value):
    if not isinstance(value, Tensor):
        value = msadapter.tensor(value, dtype=self.dtype)
    return mindspore.ops.masked_fill(self, mask, value)

@register_to_tensor(func_name="repeat")
def _repeat(self, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], Sequence):
        sizes = sizes[0]
    sizes = tuple(s if isinstance(s, int) else s.item() for s in sizes)
    return mint.tile(self, tuple(sizes))

@register_to_tensor
def view_as(self, other: Tensor):
    return self.view(other.shape)

@register_to_tensor(func_name='expand')
def _expand(self, *size):
    if len(size) == 1:
        size = size[0]
    return mint.broadcast_to(self, size)

@register_to_tensor(func_name='tile')
def _tile(self, *dims):
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        repeats = dims[0]
    else:
        repeats = dims
    return mindspore.mint.tile(self, repeats)

@register_to_tensor
def expand(self, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], Sequence):
        sizes = sizes[0]
    sizes = tuple([get_item(t) for t in sizes])
    return self.broadcast_to(sizes)

@register_to_tensor(func_name='split')
def split_(self, split_size, dim=0):
    return split(self, split_size, dim)

Tensor._ms_reshape = Tensor.reshape
@register_to_tensor(func_name='reshape')
def _reshape(self, *args, **kwargs):
    tensor = self._ms_reshape(*args, **kwargs)
    if tensor.is_contiguous():
        tensor._base = self
    return tensor

def tensor_init_wrapper(init_func):
    def wrapper(self, *args, **kwargs):
        requires_grad = kwargs.pop('requires_grad', False)
        init_func(self, *args, **kwargs)
        self._requires_grad = requires_grad
    return wrapper

def tensor_attr_del_wrapper(func):
    def wrapper(self, name):
        if name == 'grad':
            self.grad = None
        elif name == 'data':
            raise RuntimeError("Deleting tensor data is not allowed. Delete tensor instead!")
        elif name == 'requires_grad':
            raise RuntimeError("requires_grad must be a bool")
        elif name == '_grad_fn':
            raise RuntimeError("Deletion of _grad_fn not allowed. Detach tensor instead!")
        elif name == '_backward_hooks':
            raise RuntimeError("Deletion of _backwards_hooks not allowed!")
        else:
            func(self, name)
        return
    return wrapper

@register_to_tensor
def tril_(self):
    return self.copy_(self.tril())

@register_to_tensor
def bitwise_not_(self):
    return self.copy_(self.bitwise_not())

@register_to_tensor
def clamp_(self, min=None, max=None, *, out=None):
    return self.copy_(clamp(self, min, max, out=out))

@register_to_tensor
def round_(self, *, decimals=0):
    return self.copy_(round(self, decimals=decimals))


_original_is_contiguous = Tensor.is_contiguous

@register_to_tensor
def is_contiguous(self, **kwargs):
    return _original_is_contiguous(self)

ori_new_empty = Tensor.new_empty
@register_to_tensor
def new_empty(self, size,  *, dtype=None, device=None):
    if not MS28:
        size = [int(sz) for sz in size]
    return ori_new_empty(self, size, dtype=dtype, device=device)


__TENSOR_PATCH_DICT__ = dict(
        __init__=tensor_init_wrapper(Tensor.__init__),
        __delattr__ = tensor_attr_del_wrapper(Tensor.__delattr__),
        _base = None, mean = mean, clamp = clamp, is_cuda = True, is_cpu = False,
        repeat_interleave = repeat_interleave, is_sparse = False, unsqueeze = unsqueeze,
        softmax = softmax, amax=amax, ceil = ceil, cos = cos, log = log, round = round,
        sin = sin, sqrt = sqrt, square = square, sub = sub, trunc = trunc, abs = abs,
        exp = exp, floor = floor, gather = gather, matmul = matmul, mul = mul, ne = ne,
        absolute = absolute, equal = equal
)

for func_name, func in __TENSOR_PATCH_DICT__.items():
    setattr(Tensor, func_name, func)
    setattr(StubTensor, func_name, func)

StubTensor.__hash__ = Tensor.__hash__
