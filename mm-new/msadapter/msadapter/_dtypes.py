import mindspore
from mindspore.common.dtype import *


__all__ = mindspore.common.dtype.__all__
__all__.extend(['inf', 'nan', 'bool', 'int', 'long', 'float', 'cfloat', 'cdouble', 'complex32', 'SymInt', 'SymFloat', 'SymBool'])


inf = float("inf")
nan = float("nan")

bool = bool_
int = int32
long = int64
float = float32
cfloat = complex64
cdouble = complex128
complex32 = complex


class SymInt: ...

class SymFloat: ...

class SymBool: ...