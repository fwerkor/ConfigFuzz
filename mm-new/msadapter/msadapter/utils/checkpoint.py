import enum
import contextlib
import uuid
import warnings
import weakref
from collections import defaultdict
from typing import Any, Callable, ContextManager, DefaultDict, Dict, Iterable, List, Optional, Tuple
from weakref import ReferenceType
import platform

import msadapter
from msadapter.testing._internal.logging_tensor import capture_logs, LoggingTensorMode


__all__ = [
    "checkpoint",
    "checkpoint_sequential",
    "CheckpointError",
    "CheckpointFunction",
    "check_backward_validity",
    "detach_variable",
    "get_device_states",
    "set_device_states",
    "noop_context_fn",
    "set_checkpoint_early_stop",
    "DefaultDeviceType",
]

_DEFAULT_DETERMINISM_MODE = "default"


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[msadapter.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, msadapter.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError("Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)

def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, msadapter.Tensor)):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")

def _get_device_module(device="cuda"):
    device_module = getattr(msadapter, device)
    return device_module

class DefaultDeviceType:
    _default_device_type = "cuda"

    @staticmethod
    def set_device_type(device: str = "cuda"):
        DefaultDeviceType._default_device_type = device

    @staticmethod
    def get_device_type() -> str:
        return DefaultDeviceType._default_device_type

def _infer_device_type(*args):
    device_types = list({ arg.device.type for arg in args if isinstance(arg, msadapter.Tensor) and not arg.device.type == "cpu"})
    if len(device_types) > 1:
        warnings.warn(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices. "
            "Device state will only be saved for devices of a single device type, and the remaining "
            "devices will be ignored. Consequently, if any checkpointed functions involve randomness, "
            "this may result in incorrect gradients. (Note that if CUDA devices are among the devices "
            "detected, it will be prioritized; otherwise, the first device encountered will be selected.)"
        )
    if len(device_types) == 0:
        return DefaultDeviceType.get_device_type()
    elif "cuda" in device_types:
        return "cuda"
    else:
        return device_types[0]

def get_device_states(*args) -> Tuple[List[int], List[msadapter.Tensor]]:
    fwd_device_ids = list({arg.get_device() for arg in args if isinstance(arg, msadapter.Tensor) and not arg.device.type == "cpu"})
    fwd_device_states = []
    device_module = _get_device_module(_infer_device_type(*args))
    for device_id in fwd_device_ids:
        with device_module.device(device_id):
            fwd_device_states.append(device_module.get_rng_state())
    return fwd_device_ids, fwd_device_states

def set_device_states(devices, states) -> None:
    device_module = _get_device_module(_infer_device_type(*states))
    for device, state in zip(devices, states):
        with device_module.device(device):
            device_module.set_rng_state(state)

def _get_autocast_kwargs(device="cuda"):
    if device == "cuda":
        device_autocast_kwargs = {
            "enabled": msadapter.is_autocast_enabled(),
            "dtype": msadapter.get_autocast_gpu_dtype(),
            "cache_enabled": msadapter.is_autocast_cache_enabled(),
        }
    elif _supports_autocast(device):
        device_module = _get_device_module(device)
        device_autocast_kwargs = {
            "enabled": device_module.is_autocast_enabled(),
            "dtype": device_module.get_autocast_dtype(),
            "cache_enabled": msadapter.is_autocast_cache_enabled(),
        }
    else:
        device_autocast_kwargs = None
    cpu_autocast_kwargs = {
        "enabled": msadapter.is_autocast_cpu_enabled(),
        "dtype": msadapter.get_autocast_cpu_dtype(),
        "cache_enabled": msadapter.is_autocast_cache_enabled(),
    }
    return device_autocast_kwargs, cpu_autocast_kwargs

def _supports_autocast(device):
    device_module = _get_device_module(device)
    return device == "cuda" or (hasattr(device_module, "is_autocast_enabled") and hasattr(device_module, "get_autocast_dtype"))

class CheckpointFunction(msadapter.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.device = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(ctx.device)
        if preserve_rng_state:
            ctx.fwd_cpu_state = msadapter.get_rng_state()
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if msadapter.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
        ctx.save_for_backward(*tensor_inputs)
        with msadapter.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not msadapter.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs` argument.")
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors
        device_module = _get_device_module(ctx.device)
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with msadapter.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device):
            if ctx.preserve_rng_state:
                msadapter.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
            detached_inputs = detach_variable(tuple(inputs))
            device_autocast_ctx = device_module.amp.autocast(**ctx.device_autocast_kwargs) if _supports_autocast(ctx.device) else contextlib.nullcontext()
            with msadapter.enable_grad(), device_autocast_ctx, msadapter.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                outputs = ctx.run_function(*detached_inputs)
        if isinstance(outputs, msadapter.Tensor):
            outputs = (outputs,)
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if msadapter.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("none of output has requires_grad=True, this checkpoint() is not necessary")
        msadapter.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, msadapter.Tensor) else None for inp in detached_inputs)
        return (None, None) + grads

def noop_context_fn():
    return contextlib.nullcontext(), contextlib.nullcontext()

def checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
    if use_reentrant is None:
        warnings.warn(
            "msadapter.utils.checkpoint: please pass in use_reentrant=True or "
            "use_reentrant=False explicitly. The default value of use_reentrant "
            "will be updated to be False in the future. To maintain current "
            "behavior, pass use_reentrant=True. It is recommended that you use "
            "use_reentrant=False. Refer to docs for more details on the "
            "differences between the two variants."
        )
        use_reentrant = True
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError("Passing `context_fn` or `debug` is only supported when use_reentrant=False.")
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(function, preserve, context_fn, determinism_check, debug, *args, **kwargs)
        next(gen)
        ret = function(*args, **kwargs)
        try:
            next(gen)
        except StopIteration:
            return ret

def checkpoint_sequential(functions, segments, input, use_reentrant=True, **kwargs):
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))
    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input)
            return input
        return forward

    if isinstance(functions, msadapter.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint(run_function(start, end, functions), input, use_reentrant=use_reentrant, preserve_rng_state=preserve)
    return run_function(end + 1, len(functions) - 1, functions)(input)

def _internal_assert(cond):
    if not cond:
        raise AssertionError("Something went unexpectedly wrong in activation checkpoint. Please report this bug by filing an issue to Pymsadapter.")

_enable_checkpoint_early_stop = True

@contextlib.contextmanager
def set_checkpoint_early_stop(enable: bool):
    global _enable_checkpoint_early_stop
    try:
        prev = _enable_checkpoint_early_stop
        _enable_checkpoint_early_stop = enable
        yield
    finally:
        _enable_checkpoint_early_stop = prev

class _Handle:
    pass

class _Holder:
    def __init__(self):
        self.handles: Dict[int, Optional[_Handle]] = dict()

class _NoopSaveInputs(msadapter.autograd.Function):
    @staticmethod
    def forward(*args):
        return msadapter.empty((0,))

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        tensor_indices, tensors = zip(*[(i, o) for i, o in enumerate(inputs) if isinstance(o, msadapter.Tensor)])
        idx2saved_idx = {b: a for a, b in enumerate(tensor_indices)}
        args = [None if isinstance(o, msadapter.Tensor) else o for o in inputs]
        def get_args(saved_tensors):
            ret = [saved_tensors[idx2saved_idx[i]] if i in tensor_indices else o for i, o in enumerate(args)]
            return ret[1:]
        ctx.get_args = get_args
        ctx.save_for_backward(*tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise AssertionError("Did not expect to backward on this graph")

class _CheckpointFrame:
    def __init__(self, recompute_fn, early_stop, unpack_error_cb, metadata_fn):
        self.recompute_fn = recompute_fn
        self.input_saver = None
        self.weak_holders: List[ReferenceType] = []
        self.recomputed: DefaultDict[int, weakref.WeakKeyDictionary[_Handle, msadapter.Tensor]
        ] = defaultdict(weakref.WeakKeyDictionary)
        self.recomp_counter: DefaultDict[int, int] = defaultdict(int)
        self.is_recomputed: DefaultDict[int, bool] = defaultdict(bool)
        self.early_stop = early_stop
        self.metadata_fn = metadata_fn
        self.unpack_error_cb = unpack_error_cb
        self.x_metadatas = []
        self.forward_completed = False
        self.ignore_saved_mismatch = False

    def check_recomputed_tensors_match(self, gid):
        if self.ignore_saved_mismatch:
            return # TODO: we can probably make this check stricter by checking that the metadata of the first tensors still match.

        if not len(self.weak_holders) == self.recomp_counter[gid]:
            raise CheckpointError(
                "msadapter.utils.checkpoint: A different number of tensors was saved "
                "during the original forward and recomputation.\n"
                f"Number of tensors saved during forward: {len(self.weak_holders)}\n"
                f"Number of tensors saved during recomputation: {self.recomp_counter[gid]}")
        nb_meta_different = []
        for idx, weak_holder in enumerate(self.weak_holders):
            holder = weak_holder()
            if holder is None:
                continue
            _internal_assert(gid in holder.handles)
            _internal_assert(holder.handles[gid] is not None)
            _internal_assert(holder.handles[gid] in self.recomputed[gid])
            x_meta = self.x_metadatas[idx]
            recomputed_x = self.recomputed[gid][holder.handles[gid]]
            if x_meta != self.metadata_fn(recomputed_x):
                nb_meta_different.append((idx, x_meta, self.metadata_fn(recomputed_x)))

        if len(nb_meta_different) > 0:
            mismatched_tensors = ""
            for idx, x_meta, recomputed_meta in nb_meta_different:
                mismatched_tensors += (f"tensor at position {idx}:\nsaved metadata: {x_meta}\nrecomputed metadata: {recomputed_meta}\n")
            raise CheckpointError("msadapter.utils.checkpoint: Recomputed values for the following tensors "
                "have different metadata than during the forward pass.\n{mismatched_tensors}")

_checkpoint_error_template = """ \
An error happened while unpacking tensors; dumping logs of latest computation
because you passed `debug=True` to `msadapter.utils.checkpoint.checkpoint()`.
Scroll all the way down for guidance on how to navigate these logs.

+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        1. Stack traces of the operators that ran in the original forward     |
+------------------------------------------------------------------------------+

{forward_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        2. Stack traces of the operators that ran during recomputation        |
+------------------------------------------------------------------------------+

{recompute_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|       3. Log of operators in the original forward and recomputation          |
+------------------------------------------------------------------------------+
(Scroll up to correlate stack traces with each operation listed below. This
 helps identify their source in the code.)

IMPORTANT: Differences in "detach" calls between the original forward and the
           recomputation are expected. They are introduced by the checkpointing
           mechanism and can be ignored.

Operations executed during the original forward:

{forward_ops}

Operations executed during recomputation:

{recompute_ops}

+------------------------------------------------------------------------------+
 ERROR: Detected non-determinism while running activation checkpointing

 You are seeing this error because you passed `debug=True` to checkpoint and
 tensors to be saved during the original forward and differ between those saved
 during recomputation. This can happen if different operators were ran in the
 original forward and in the recomputation.

 To identify where the mismatch may be coming from, you can do the following:

 1) Compare the operators ran during original forward and recomputation to
    see where they differ. These operators are printed above in the order they
    were executed.

 2) Review the stack trace for each operator to locate its invocation source.
    Each operator's stack trace is printed in their execution order.

 Note that the logs can be quite long. Here's how they are structured:
 (Tip: you can Ctrl-f for these headers)

 1. Stack traces of the operators that ran in the original forward
 2. Stack traces of the operators that ran during recomputation
 3. Log of operators in the original forward and recomputation
 4. Error message                                             <--- You are here
--------------------------------------------------------------------------------
"""

class CheckpointError(RuntimeError):
    pass

def _default_meta_extractor(x: msadapter.Tensor) -> Dict[str, Any]:
    return {"shape": x.shape, "dtype": x.dtype, "device": x.device}

_allowed_determinism_checks_to_fns: Dict[str, Callable[[msadapter.Tensor], Any]] = {_DEFAULT_DETERMINISM_MODE: _default_meta_extractor, "none": lambda _: None,}

class _StopRecomputationError(Exception):
    pass

class _recomputation_hook(msadapter.autograd.graph.saved_tensors_hooks):
    def __init__(self, target_frame_ref: ReferenceType, gid: int):
        def pack_hook(x):
            target_frame = target_frame_ref()
            assert target_frame is not None  # appease mypy
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1

            if recomp_idx >= len(target_frame.weak_holders):
                assert not target_frame.early_stop
                if not target_frame.forward_completed:
                    target_frame.ignore_saved_mismatch = True
                    return x.detach()
                raise CheckpointError("msadapter.utils.checkpoint: trying to save more tensors during recomputation than during the original forward pass.")

            holder = target_frame.weak_holders[recomp_idx]()
            if holder is not None:
                _internal_assert(holder.handles.get(gid, None) is None)
                holder.handles[gid] = _Handle()
                target_frame.recomputed[gid][holder.handles[gid]] = x.detach()
            if target_frame.early_stop and target_frame.recomp_counter[gid] == len(
                target_frame.weak_holders
            ):
                raise _StopRecomputationError()
            return x.detach()

        def unpack_hook(x):
            return x
        super().__init__(pack_hook, unpack_hook)

class _checkpoint_hook(msadapter.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame):
        def pack_hook(x):
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            if frame.metadata_fn is not None:
                with msadapter.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            gid = -1
            if gid == -1:
                gid = int(uuid.uuid4())
            if not frame.is_recomputed[gid]:
                ctx = frame.input_saver.grad_fn
                args = ctx.get_args(ctx.saved_tensors)
                try:
                    with _recomputation_hook(
                        weakref.ref(frame), gid
                    ), msadapter.autograd.enable_grad():
                        frame.recompute_fn(*args)
                except _StopRecomputationError:
                    pass
                frame.is_recomputed[gid] = True
                frame.check_recomputed_tensors_match(gid)
            _internal_assert(gid in holder.handles)
            if holder.handles[gid] is None:
                raise CheckpointError(
                    "msadapter.utils.checkpoint: Unpack is being triggered for a tensor that was already "
                    "unpacked once. If you are calling ctx.saved_tensors in backward, make sure to do "
                    "so only once. Otherwise please open an issue with details on your use case."
                )
            _internal_assert(holder.handles[gid] in frame.recomputed[gid])
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret

        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except CheckpointError as e:
                    frame.unpack_error_cb(e)
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            super().__init__(pack_hook, unpack_hook)

def _get_debug_context_and_cb() -> Tuple[Callable[[], Any], Callable[[CheckpointError], None]]:
    cpp_tb = platform.machine() == 'x86_64' and platform.system() == 'Linux'

    class CaptureLogs:
        def __init__(self):
            self.logs = None
            self.tbs = None

        def get_context_manager(self):
            @contextlib.contextmanager
            def logging_mode():
                with LoggingTensorMode(), \
                     capture_logs(True, python_tb=True, script_tb=True, cpp_tb=cpp_tb) as logs_and_tb:
                    self.logs, self.tbs = logs_and_tb
                    yield logs_and_tb
            return logging_mode()

    capture_logs_fwd = CaptureLogs()
    capture_logs_recompute = CaptureLogs()

    def unpack_error_cb(e: CheckpointError):
        def get_str_tb(label, capture_logs):
            out = ""
            total_len = len(capture_logs.logs)
            for i, (log, tb) in enumerate(zip(capture_logs.logs, capture_logs.tbs)):
                out += f"{log}   ({i + 1} of {total_len} in {label})\n\n"
                found_torch_dispatch = False
                for line in tb:
                    is_torch_dispatch = line['name'] == '__torch_dispatch__'
                    if not found_torch_dispatch and not is_torch_dispatch:
                        continue
                    elif is_torch_dispatch:
                        found_torch_dispatch = True
                        continue
                    out += f"{line['filename']}:{line['line']}:{line['name']}\n"
                out += "\n\n"
            return out
        assert capture_logs_fwd.logs is not None
        assert capture_logs_recompute.logs is not None
        raise CheckpointError(_checkpoint_error_template.format(
                forward_traces=get_str_tb("original", capture_logs_fwd),
                recompute_traces=get_str_tb("recompute", capture_logs_recompute),
                forward_ops="\n".join(capture_logs_fwd.logs),
                recompute_ops="\n".join(capture_logs_recompute.logs))) from e

    def context_fn():
        return capture_logs_fwd.get_context_manager(), capture_logs_recompute.get_context_manager()

    return context_fn, unpack_error_cb

def _checkpoint_without_reentrant_generator(
    fn,
    preserve_rng_state=True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    *args,
    **kwargs
):
    unpack_error_cb = None
    if debug:
        if context_fn != noop_context_fn:
            raise ValueError("debug=True is incompatible with non-default context_fn")
        context_fn, unpack_error_cb = _get_debug_context_and_cb()
    if determinism_check in _allowed_determinism_checks_to_fns:
        metadata_fn = _allowed_determinism_checks_to_fns[determinism_check]
    else:
        raise ValueError(f"determinism_check should be one of {list(_allowed_determinism_checks_to_fns.keys())}, but got {determinism_check}")
    device = _infer_device_type(*args)
    device_module = _get_device_module(device)
    forward_context, recompute_context = context_fn()
    device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device=device)

    if preserve_rng_state:
        fwd_cpu_state = msadapter.get_rng_state()
        had_device_in_fwd = False
        if getattr(device_module, "_initialized", False):
            had_device_in_fwd = True
            fwd_devices, fwd_device_states = get_device_states(*args)

    def recompute_fn(*inputs):
        kwargs, *args = inputs
        rng_devices = []
        if preserve_rng_state and had_device_in_fwd:
            rng_devices = fwd_devices
        with msadapter.random.fork_rng(devices=rng_devices, enabled=preserve_rng_state, device_type=device):
            if preserve_rng_state:
                msadapter.set_rng_state(fwd_cpu_state)
                if had_device_in_fwd:
                    set_device_states(fwd_devices, fwd_device_states)
            device_autocast_ctx = device_module.amp.autocast(**device_autocast_kwargs) if _supports_autocast(device) else contextlib.nullcontext()
            with device_autocast_ctx, msadapter.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:
                fn(*args, **kwargs)

    new_frame = _CheckpointFrame(recompute_fn, _enable_checkpoint_early_stop, unpack_error_cb, metadata_fn)
    dummy = msadapter.empty((0,), requires_grad=True)
    new_frame.input_saver = _NoopSaveInputs.apply(dummy, kwargs, *args)

    if new_frame.input_saver.grad_fn is None:
        yield
        return

    with _checkpoint_hook(new_frame), forward_context:
        yield
    new_frame.forward_completed = True

    if getattr(device_module, "_initialized", False) and \
       preserve_rng_state and not had_device_in_fwd:
        raise RuntimeError("PyTorch's device state was initialized in the forward pass of a Checkpoint, which is not allowed. Please open an issue if you need this feature.")
    return

class SelectiveCheckpointContext:
    def __init__(self, *, is_recompute):
        self.is_recompute = is_recompute

class CheckpointPolicy(enum.Enum):
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3

def _policy_from_bool(b):
    return CheckpointPolicy.MUST_SAVE if b else CheckpointPolicy.PREFER_RECOMPUTE
