import sys
import traceback
from functools import reduce
import operator

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common import np_dtype
from mindspore._c_expression import TensorPy
from .ops import empty, narrow
from .configs import support_bf16


element_size_map = {
    mindspore.float16: 2,
    mindspore.float32: 3,
    mindspore.bfloat16: 2,
    mindspore.int64: 4,
    mindspore.uint8: 1,
    mindspore.int8: 1,
    mindspore.bool_: 1
}

def _bf16():
    if not hasattr(_bf16, 'bf16'):
        if support_bf16() and hasattr(np_dtype, 'bfloat16'):
            _bf16.bf16 = np_dtype.bfloat16
        else:
            import ml_dtypes
            _bf16.bf16 = ml_dtypes.bfloat16
    return _bf16.bf16

def _element_size(dtype):
    return element_size_map[dtype]

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    tensors = [tensor.view(-1) for tensor in tensors]
    return mindspore.mint.cat(tensors)


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        if numel == 0:
            outputs.append(empty(0, dtype=flat.dtype))
        else:
            outputs.append(narrow(flat, 0, offset, numel).view(tensor.shape))
            offset += numel
    return outputs

def _is_contiguous_by_shape_stride(shape, stride):
    if len(shape) != len(stride):
        raise ValueError("shape and stride must have the same length.")

    if len(shape) == 0:
        return True

    expected_stride = 1
    for i in range(len(shape)-1, -1, -1):
        if stride[i] != expected_stride:
            return False
        expected_stride *= shape[i]

    return True

def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks=None, metadata=None):
    '''Rebuilds a tensor based on the provided parameters.
    
    Args:
        storage (ndarray): The storage array from which the tensor is created.
        storage_offset (int): The offset in the storage array from where the tensor data starts.
        size (tuple): The size of the tensor.
        stride (tuple or None): The stride of the tensor, or None if not applicable.
        requires_grad (bool): Indicates if the tensor requires gradient computation.
        backward_hooks (list): A list of backward hooks for the tensor.
        metadata (Any, optional): Additional metadata associated with the tensor.
    
    Returns:
        None: This function does not have a return value.
    
    Raises:
        None: This function does not raise any exceptions.
    '''
    if isinstance(storage, TensorPy):
        return Tensor(storage)

    if size == ():
        num_elemets = 1
    else:
        num_elemets = reduce(operator.mul, size)

    array = storage

    if num_elemets == storage.size and _is_contiguous_by_shape_stride(size, stride):
        if array.dtype == _bf16() and not support_bf16():
            array = array.astype(np.float16)
        array = array.reshape(size)
        if isinstance(array, np.memmap):
            array = array.copy()
        return Tensor.from_numpy(array)

    target_dtype = storage.dtype
    if array.dtype == _bf16():
        if support_bf16():
            array = array.view(np.float16)
        else:
            array = array.astype(np.float16)
            target_dtype = array.dtype

    itemsize = array.dtype.itemsize
    np_offset = storage_offset * itemsize
    np_strides = tuple(x * itemsize for x in stride)
    array = np.ndarray(shape=size, dtype=target_dtype, buffer=array.data, offset=np_offset, strides=np_strides)
    return Tensor.from_numpy(array)


def dtype_to_nptype(dtype):
    """
    Convert MindSpore dtype to numpy data type.
    """
    dtype_nptype_dict = {
        mindspore.bool_: np.bool_,
        mindspore.int8: np.int8,
        mindspore.int16: np.int16,
        mindspore.int32: np.int32,
        mindspore.int64: np.int64,
        mindspore.uint8: np.uint8,
        mindspore.uint16: np.uint16,
        mindspore.uint32: np.uint32,
        mindspore.uint64: np.uint64,
        mindspore.float16: np.float16,
        mindspore.float32: np.float32,
        mindspore.float64: np.float64,
        mindspore.complex64: np.complex64,
        mindspore.complex128: np.complex128,
    }
    if dtype == mindspore.bfloat16:
        if hasattr(np_dtype, 'bfloat16'):
            return np_dtype.bfloat16
        import ml_dtypes
        return ml_dtypes.bfloat16
    return dtype_nptype_dict[dtype]

def _get_device_module(device_type: str):
    pass

class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""

    def __repr__(self):
        return self

class ExceptionWrapper:
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        # Format a message such as: "Caught ValueError in DataLoader worker
        # process 2. Original Traceback:", followed by the traceback.
        msg = f"Caught {self.exc_type.__name__} {self.where}.\nOriginal {self.exc_msg}"
        if self.exc_type == KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            # Some exceptions have first argument as non-str but explicitly
            # have message field
            raise self.exc_type(message=msg)
        try:
            exception = self.exc_type(msg)
        except Exception:
            # If the exception takes multiple arguments or otherwise can't
            # be constructed, don't try to instantiate since we don't know how to
            raise RuntimeError(msg) from None
        raise exception
