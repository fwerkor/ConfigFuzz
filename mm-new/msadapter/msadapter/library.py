"""This is a temp mock class for adapting MindSpeed op_builder import"""
import sys
from functools import wraps

from typing import (Callable, Optional, Union)

class Library:
    def __init__(self, foo, bar):
        self.name = foo

    def define(self, foo):
        pass

_op_identifier = Union[str,]

def register_fake(
    op: _op_identifier,
    func: Optional[Callable] = None,
    /,
    *,
    lib: Optional[Library] = None,
    _stacklevel: int = 1,
    allow_override: bool = False,
):
    if not isinstance(op, (str, )):
        raise ValueError(f"register_fake({op}): got unexpected type for op: {type(op)}")

    def register(func):
        return func

    if func is None:
        return register
    return register(func)

def impl(as_library, name, bar):
    def decorator(func):
        full_path = f"msadapter.ops.{as_library.name}.{name}.default"
        parts = full_path.split('.')

        current_module = None
        current_path = []

        for part in parts[:-1]:
            current_path.append(part)
            module_name = '.'.join(current_path)

            if module_name not in sys.modules:
                module = type(sys)(module_name)
                sys.modules[module_name] = module
                if current_module is not None:
                    setattr(current_module, part, module)
            
            current_module = sys.modules[module_name]
        
        func_name = parts[-1]
        setattr(current_module, func_name, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
