"""tensor op"""
import mindspore
from mindspore._c_expression import typing # pylint: disable=no-name-in-module, import-error

def is_tensor(input):
    return isinstance(input, mindspore.Tensor)

def is_floating_point(input):
    if is_tensor(input):
        return input.is_floating_point()

    raise TypeError(f"is_floating_point(): argument 'input' (position 1) must be Tensor, not '{type(input).__name__}')")

def is_complex(input):
    if is_tensor(input):
        return input.is_complex()

    raise TypeError(f"is_complex(): argument 'input' (position 1) must be Tensor, not '{type(input).__name__}')")

def numel(input):
    return input.numel()

def as_tensor(data, dtype=None, **kwarg):
    return mindspore.Tensor(data, dtype)

__all__ = ['as_tensor', 'is_tensor', 'is_floating_point', 'is_complex', 'numel']