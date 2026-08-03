"""core module"""
import msadapter
from mindspore.common.api import _pynative_executor
from msadapter.utils._contextlib import _NoParamDecoratorContextManager, _DecoratorContextManager
from msadapter.configs import MS27
from typing import Any, Callable, TypeVar


FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

class no_grad(_NoParamDecoratorContextManager):
    def __init__(self) -> None:
        super().__init__()
        self.prev_state = False
        self.prev_recompute = False

    def __enter__(self):
        self.prev_recompute = msadapter.autograd.recompute_instance.get_recompute()
        if MS27:
            msadapter.autograd.recompute_instance.set_recompute(True)
        self.prev_state = _pynative_executor.enable_grad()
        _pynative_executor.set_enable_grad(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pynative_executor.set_enable_grad(self.prev_state)
        if MS27:
            if self.prev_recompute:
                return
            msadapter.autograd.recompute_instance.set_recompute(False)

class enable_grad(_NoParamDecoratorContextManager):
    def __enter__(self):
        self.prev_state = _pynative_executor.enable_grad()
        _pynative_executor.set_enable_grad(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pynative_executor.set_enable_grad(self.prev_state)

class set_grad_enabled(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None:
        self.prev = msadapter.is_grad_enabled()
        self.mode = mode
        _pynative_executor.set_enable_grad(mode)

    def __call__(self, orig_func: F) -> F:
        _pynative_executor.set_enable_grad(self.prev)
        return super().__call__(orig_func)

    def __enter__(self) -> None:
        _pynative_executor.set_enable_grad(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _pynative_executor.set_enable_grad(self.prev)

    def __str__(self) -> str:
        return f"{msadapter.typename(self)}(mode={self.mode})"

    def __repr__(self) -> str:
        return str(self)

    def clone(self) -> "set_grad_enabled":
        return self.__class__(self.mode)

class inference_mode(_DecoratorContextManager):
    def __init__(self, mode: bool = True) -> None:
        super().__init__()
        self.mode = mode
        self.prev_state = False

    def __new__(cls, mode: bool = True):
        if isinstance(mode, bool):
            return super().__new__(cls)
        return cls()(mode)

    def __enter__(self) -> None:
        self.prev_state = _pynative_executor.enable_grad()
        _pynative_executor.set_enable_grad(not self.mode)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _pynative_executor.set_enable_grad(self.prev_state)

    def clone(self) -> "inference_mode":
        return self.__class__(self.mode)

class set_multithreading_enabled(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None:
        self.prev = msadapter._C._is_multithreading_enabled()
        msadapter._C._set_multithreading_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        msadapter._C._set_multithreading_enabled(self.prev)

    def clone(self) -> "set_multithreading_enabled":
        return self.__class__(self.mode)