import warnings
from collections import deque
from msadapter import Tensor


_is_in_torch_dispatch_mode = False
_is_in_non_infra_torch_dispatch_mode = False

def is_traceable_wrapper_subclass(t: object):
    is_subclass = isinstance(t, Tensor) and type(t) != Tensor
    return (
        is_subclass
        and hasattr(t, "__tensor_flatten__")
        and hasattr(t, "__tensor_unflatten__")
    )

def is_in_torch_dispatch_mode(include_infra_modes=True) -> bool:
    return (_is_in_torch_dispatch_mode if include_infra_modes else _is_in_non_infra_torch_dispatch_mode)

class TorchDispatchMode:
    supports_higher_order_operators = False

    def __init__(self, _dispatch_key=None):
        if _dispatch_key is not None:
            self.__dict__["_dispatch_key"] = _dispatch_key

        self.old_dispatch_mode_flags: deque[bool] = deque()
        self.old_non_infra_dispatch_mode_flags: deque[bool] = deque()

    def _lazy_init_old_dispatch_mode_flags(self):
        if not hasattr(self, "old_dispatch_mode_flags"):
            self.old_dispatch_mode_flags: deque[bool] = deque()  # type: ignore[no-redef]

        if not hasattr(self, "old_non_infra_dispatch_mode_flags"):
            self.old_non_infra_dispatch_mode_flags: deque[bool] = deque()  # type: ignore[no-redef]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError

    def __enter__(self):
        warnings.warn(f"TorchDispatchMode is just a placeholder in msadapter and works not effect.")
        global _is_in_torch_dispatch_mode
        global _is_in_non_infra_torch_dispatch_mode
        self._lazy_init_old_dispatch_mode_flags()
        self.old_dispatch_mode_flags.append(_is_in_torch_dispatch_mode)
        _is_in_torch_dispatch_mode = True
        self.old_non_infra_dispatch_mode_flags.append(_is_in_non_infra_torch_dispatch_mode)
        _is_in_non_infra_torch_dispatch_mode = (_is_in_non_infra_torch_dispatch_mode or not self.is_infra_mode())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mb_dk_or_mode_key = self.__dict__.get("_dispatch_key", None)
        if mb_dk_or_mode_key is None:
            mb_dk_or_mode_key = self.__dict__.get("_mode_key", None)
        global _is_in_torch_dispatch_mode
        _is_in_torch_dispatch_mode = self.old_dispatch_mode_flags.pop()
        global _is_in_non_infra_torch_dispatch_mode
        _is_in_non_infra_torch_dispatch_mode = (self.old_non_infra_dispatch_mode_flags.pop())
