from mindspore import ops, mint
from ..ops._inner import call_ms_func
from ..configs import use_pyboost

__all__ = ['norm', 'vector_norm']


def norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None):
    if use_pyboost():
        return call_ms_func(mint.linalg.norm, A, ord, dim, keepdim, dtype=dtype, out=out)
    return call_ms_func(ops.norm, A, ord, dim, keepdim, dtype=dtype, out=out)


def vector_norm(x, ord=2, dim=None, keepdim=False, *, out=None, dtype=None):
    return call_ms_func(mint.linalg.vector_norm, x, ord, dim, keepdim, dtype=dtype, out=out)
