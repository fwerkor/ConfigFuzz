from . import ge
from functools import wraps

def register_fx_node_ge_converter(foo):
    def f(func):
        return func
    return f
