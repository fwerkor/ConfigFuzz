import operator
from functools import reduce


def _get_tuple_numel(input):
    if input == ():
        return 1
    return reduce(operator.mul, list(input))


class Size(tuple):
    def __new__(cls, shape=(), check_shape=True):
        _shape = shape

        if check_shape:
            for index, item in enumerate(_shape):
                if not isinstance(item, int):
                    raise TypeError(f"msadapter.Size() takes an iterable of 'int' (item {index} is '{type(item).__name__}')")

        return tuple.__new__(Size, _shape)

    def __getitem__(self, key, /):
        if isinstance(key, slice):
            return Size(super().__getitem__(key))
        return super().__getitem__(key)

    def __reduce__(self):
        return (self.__class__, (tuple(self), ))

    def numel(self):
        return _get_tuple_numel(self)

    def __repr__(self):
        return "msadapter.Size(" + str(list(self)) + ")"
