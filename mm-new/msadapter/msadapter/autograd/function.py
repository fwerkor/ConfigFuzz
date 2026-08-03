"""functional autograd"""
import warnings
import functools
from typing import Any, AnyStr, overload, Tuple, Union
from collections.abc import Generator

import mindspore
from mindspore.ops.composite import GradOperation
from mindspore.ops import stop_gradient
from mindspore.common.api import _pynative_executor
from mindspore._c_expression import Cell_
from .grad_mode import no_grad

import msadapter
from msadapter import Tensor
try:
    from mindspore import _Function
except:
    Function = None


class RecomputeFlag:
    def __init__(self):
        self.recompute = False

    def set_recompute(self, flag):
        self.recompute = flag

    def get_recompute(self):
        return self.recompute

recompute_instance = RecomputeFlag()

if _Function is not None:
    class Function(_Function):
        __INVALID_ATTRS__ = {"register_hook", "_register_hook_dict", "next_functions", "name"}
        def __getattr__(self, name: str) -> Any:
            if name == "saved_variables":
                warnings.warn("'saved_variables' is deprecated; use 'saved_tensors'", DeprecationWarning)
                return self.saved_tensors
            elif name in self.__INVALID_ATTRS__:
                raise RuntimeError(f"Attribute '{name}' is invalid for this instance of FunctionBase_. "
                                   f"Accessing this attribute directly on an instance of autograd.Function is a "
                                   f"legacy access pattern that is no longer supported. For examples on how to use "
                                   f"new-style autograd functions, see https://pymsadapter.org/docs/stable/autograd.html#msadapter.autograd.Function")
            elif name == "metadata":
                raise RuntimeError("You attempted to access the anomaly metadata of a custom autograd function but the underlying PyNode "
                                   "has already been deallocated.  The most likely reason this occurred is because you assigned x.grad_fn "
                                   "to a local variable and then let the original variable get deallocated.  Don't do that!  If you really "
                                   "have no way of restructuring your code so this is the case, please file an issue reporting that you are "
                                   "affected by this.")
            return getattr(self.real, name)

        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                "Legacy autograd function with non-static forward method is deprecated. "
                "Please use new-style autograd function with static forward method. "
                "(Example: https://pymsadapter.org/docs/stable/autograd.html#msadapter.autograd.Function)"
            )
        generate_vmap_rule = False

        @staticmethod
        def vmap(info, in_dims, *args):
            raise NotImplementedError("To use autograd.Function with vmap, you must either override the vmap staticmethod or set generate_vmap_rule=True.")

        @staticmethod
        def forward(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("You must implement the forward function for custom autograd.Function.")

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            raise NotImplementedError("setup_context is not implemented.")

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            raise NotImplementedError("You must implement either the backward or vjp method for your custom autograd.Function to use it with backward mode AD.")

        vjp = backward

        @staticmethod
        def jvp(ctx: Any, *grad_inputs: Any) -> Any:

            raise NotImplementedError("You must implement the jvp function for custom autograd.Function to use it with forward mode AD.")

else:
    class Function(Cell_):
        generate_vmap_rule = False
        def __init__(self):
            super().__init__(str(self.__class__)[8:-2])
            self.saved_tensors = []
            self.used_bprop_inputs = []

        def save_for_backward(self, *args):
            if isinstance(args, tuple):
                self.saved_tensors.extend(list(args))
            else:
                self.saved_tensors.append(args)

        @staticmethod
        def forward(ctx, *args, **kwargs):
            raise NotImplementedError

        @staticmethod
        def backward(ctx, *args, **kwargs):
            raise NotImplementedError

        def construct(self, *args, **kwargs):
            self.needs_input_grad = [input_.requires_grad if hasattr(input_, 'requires_grad') else False for input_ in args]
            args = (self,) + args
            return self.forward(*args, **kwargs)

        def bprop(self, *args, **kwargs):
            args = (args[-1],)
            args = (self,) + args
            return self.backward(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            with no_grad():
                output = self.construct(*args, **kwargs)
            if _pynative_executor.requires_grad():
                _pynative_executor.call_custom_bprop(self, output, *args, **kwargs)
            return output

        @classmethod
        def apply(cls, *args, **kwargs):
            return cls()(*args, **kwargs)

class InplaceFunction(Function):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

class DelayedError:
    def __init__(self, msg: AnyStr, num_inputs: int) -> None:
        self.msg = msg
        self.num_inputs = num_inputs

    @overload
    def __call__(self, i0: Tensor) -> Tensor: ...

    @overload
    def __call__(self, *args: Tensor) -> Tuple[Tensor, ...]: ...

    def __call__(self, *args: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        raise RuntimeError(f"DelayedError triggered: {self.msg}")

def once_differentiable(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args):
        with msadapter.no_grad():
            outputs = fn(ctx, *args)
        if not msadapter.is_grad_enabled():
            return outputs
        requires_grad = any(isinstance(arg, msadapter.Tensor) and arg.requires_grad for arg in args)
        if not requires_grad:
            return outputs
        if not isinstance(outputs, tuple): # TODO: for check
            outputs = (outputs,)
        err_fn = DelayedError(b"trying to differentiate twice a function that was marked with @once_differentiable", len(outputs))
        def fake_requires_grad(var):
            if var is not None:
                var = var.detach()
                var.requires_grad = True
            return var
        return err_fn(*[fake_requires_grad(v) for v in outputs])
    return wrapper