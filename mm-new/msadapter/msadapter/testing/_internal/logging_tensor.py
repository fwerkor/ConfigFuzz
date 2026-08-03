# mypy: ignore-errors
import msadapter
from msadapter.utils._pytree import tree_map
from typing import Iterator, List, Optional
import logging
import contextlib
import itertools
from msadapter.utils._python_dispatch import TorchDispatchMode
from msadapter.utils.weak import WeakTensorKeyDictionary
from msadapter.utils._dtype_abbrs import dtype_abbrs as _dtype_abbrs
import functools


logger = logging.getLogger("LoggingTensor")


# TODO: TensorBase should work
class LoggingTensor(msadapter.Tensor):
    elem: msadapter.Tensor
    __slots__ = ['elem']
    context = contextlib.nullcontext

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = msadapter.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=kwargs.get("requires_grad", False)
        )
        r.elem = elem.detach() if r.requires_grad else elem
        return r

    def __repr__(self):
        return super().__repr__(tensor_contents=f"{self.elem}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            return cls(e) if isinstance(e, msadapter.Tensor) else e

        with cls.context():
            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)  # noqa: G004
        return rs

class LoggingTensorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)  # noqa: G004
        return rs

class LoggingTensorReentrant(LoggingTensor):
    context = msadapter.overrides.enable_reentrant_dispatch

class LoggingTensorHandler(logging.Handler):
    def __init__(
            self, log_list: List[str], use_shortid_for_all_tensors: bool,
            with_type: bool, tracebacks_list: Optional[List]) -> None:
        logging.Handler.__init__(self)
        self.log_list = log_list
        self.use_shortid_for_all_tensors = use_shortid_for_all_tensors
        self.tracebacks_list = tracebacks_list
        self.memo = WeakTensorKeyDictionary()
        self.next_id = 0
        self.with_type = with_type

    def _shortid(self, t: msadapter.Tensor) -> int:
        if t not in self.memo:
            self.memo[t] = self.next_id
            self.next_id += 1
        return self.memo[t]

    def _fmt(self, a: object, with_type: bool = False) -> str:
        cond_cls = msadapter.Tensor if self.use_shortid_for_all_tensors else LoggingTensor
        if isinstance(a, cond_cls):
            maybe_type = ""
            if with_type and self.with_type:
                maybe_type = f": {_dtype_abbrs[a.dtype]}[{', '.join(map(str, a.shape))}]"
            x = f"${self._shortid(a)}{maybe_type}"
            return x
        else:
            return repr(a)

    def emit(self, record):
        fmt_args = ", ".join(
            itertools.chain(
                (str(tree_map(self._fmt, a)) for a in record.args[0]),
                (f"{k}={str(tree_map(self._fmt, v))}" for k, v in record.args[1].items()),
            )
        )
        fmt_rets = tree_map(functools.partial(self._fmt, with_type=True), record.args[2])
        self.log_list.append(f'{fmt_rets} = {record.msg}({fmt_args})')
        if self.tracebacks_list is not None:
            self.tracebacks_list.append(record.traceback)

def log_input(name: str, var: object) -> None:
    logger.info("input", (name,), {}, var)  # noqa: PLE1205

class GatherTraceback(logging.Filter):
    def __init__(self, python=True, script=True, cpp=False):
        self.python = python
        self.script = script
        self.cpp = cpp

    def filter(self, record):
        return True

@contextlib.contextmanager
def capture_logs(is_mode=False, python_tb=False, script_tb=False, cpp_tb=False) -> Iterator[List[str]]:
    collect_traceback = python_tb or script_tb or cpp_tb
    log_list: List[str] = []
    tracebacks_list: List[str] = []
    handler = LoggingTensorHandler(log_list, with_type=True,
        use_shortid_for_all_tensors=is_mode, tracebacks_list=tracebacks_list if collect_traceback else None)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if collect_traceback:
        logger.addFilter(GatherTraceback(python=python_tb, script=script_tb, cpp=cpp_tb))
    try:
        if collect_traceback:
            yield log_list, tracebacks_list
        else:
            yield log_list
    finally:
        tracebacks_list.clear()
        logger.removeHandler(handler)

@contextlib.contextmanager
def capture_logs_with_logging_tensor_mode(python_tb=False, script_tb=False, cpp_tb=False):
    with LoggingTensorMode(), capture_logs(True, python_tb, script_tb, cpp_tb) as logs:
        yield logs