import __future__  # noqa: F404
import contextlib
import msadapter


def has_torch_function(objs):
    return any(hasattr(obj, '__torch_function__') and type(obj).__torch_function__ is not msadapter.Tensor.__torch_function__ for obj in objs)

def handle_torch_function(func, args, kwargs=None):
    kwargs = kwargs or {}
    overloaded_types = set(type(arg) for arg in args if hasattr(arg, '__torch_function__'))
    if not overloaded_types:
        return func(*args, **kwargs)
    overloaded_types = sorted(overloaded_types, key=lambda cls: cls.__mro__)
    return overloaded_types[0].__torch_function__(func, tuple(overloaded_types), args, kwargs)

def is_tensor_like(inp):
    return type(inp) is msadapter.Tensor or hasattr(type(inp), "__torch_function__")

@contextlib.contextmanager
def enable_reentrant_dispatch():
    try:
        yield
    finally:
        pass