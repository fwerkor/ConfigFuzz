# mypy: allow-untyped-defs
import msadapter


__all__ = ["vjp", "jvp", "hvp", "vhp"]


def _as_tuple_nocheck(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)

def _as_tuple(inp, arg_name=None, fn_name=None):
    if arg_name is None and fn_name is None:
        return _as_tuple_nocheck(inp)
    is_inp_tuple = True
    if not isinstance(inp, tuple):
        inp = (inp,)
        is_inp_tuple = False
    for i, el in enumerate(inp):
        if not isinstance(el, msadapter.Tensor):
            if is_inp_tuple:
                raise TypeError(f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the value at index {i} has type {type(el)}.")
            else:
                raise TypeError(f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the given {arg_name} has type {type(el)}.")
    return is_inp_tuple, inp

def _tuple_postprocess(res, to_unpack):
    if isinstance(to_unpack, tuple):
        assert len(to_unpack) == 2
        if not to_unpack[1]:
            res = tuple(el[0] for el in res)
        if not to_unpack[0]:
            res = res[0]
    else:
        if not to_unpack:
            res = res[0]
    return res

def _grad_preprocess(inputs, create_graph, need_graph):
    res = []
    for inp in inputs:
        if create_graph and inp.requires_grad:
            if not inp.is_sparse:
                res.append(inp.view_as(inp))
            else:
                res.append(inp.clone())
        else:
            res.append(inp.detach().requires_grad_(need_graph))
    return tuple(res)

def _grad_postprocess(inputs, create_graph):
    if isinstance(inputs[0], msadapter.Tensor):
        if not create_graph:
            return tuple(inp.detach() for inp in inputs)
        else:
            return inputs
    else:
        return tuple(_grad_postprocess(inp, create_graph) for inp in inputs)

def _validate_v(v, other, is_other_tuple):
    if len(other) != len(v):
        if is_other_tuple:
            raise RuntimeError(f"v is a tuple of invalid length: should be {len(other)} but got {len(v)}.")
        else:
            raise RuntimeError("The given v should contain a single Tensor.")

    for idx, (el_v, el_other) in enumerate(zip(v, other)):
        if el_v.size() != el_other.size():
            prepend = ""
            if is_other_tuple:
                prepend = f"Entry {idx} in "
            raise RuntimeError(f"{prepend}v has invalid size: should be {el_other.size()} but got {el_v.size()}.")

def _check_requires_grad(inputs, input_type, strict):
    if not strict:
        return
    if input_type not in ["outputs", "grad_inputs", "jacobian", "hessian"]:
        raise RuntimeError("Invalid input_type to _check_requires_grad")
    for i, inp in enumerate(inputs):
        if inp is None:
            raise RuntimeError(f"The output of the user-provided function is independent of input {i}. This is not allowed in strict mode.")
        if not inp.requires_grad:
            if input_type == "hessian":
                raise RuntimeError(
                    f"The hessian of the user-provided function with respect to input {i}"
                    " is independent of the input. This is not allowed in strict mode."
                    " You should ensure that your function is thrice differentiable and that"
                    " the hessian depends on the inputs.")
            elif input_type == "jacobian":
                raise RuntimeError(
                    "While computing the hessian, found that the jacobian of the user-provided"
                    f" function with respect to input {i} is independent of the input. This is not"
                    " allowed in strict mode. You should ensure that your function is twice"
                    " differentiable and that the jacobian depends on the inputs (this would be"
                    " violated by a linear function for example).")
            elif input_type == "grad_inputs":
                raise RuntimeError(
                    f"The gradient with respect to input {i} is independent of the inputs of the"
                    " user-provided function. This is not allowed in strict mode.")
            else:
                raise RuntimeError(
                    f"Output {i} of the user-provided function does not require gradients."
                    " The outputs must be computed in a differentiable manner from the input"
                    " when running in strict mode.")

def _autograd_grad(
    outputs,
    inputs,
    grad_outputs=None,
    create_graph=False,
    retain_graph=None,
    is_grads_batched=False,
):
    assert isinstance(outputs, tuple)
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    assert isinstance(grad_outputs, tuple)
    assert len(outputs) == len(grad_outputs)
    new_outputs: tuple[msadapter.Tensor, ...] = ()
    new_grad_outputs: tuple[msadapter.Tensor, ...] = ()
    for out, grad_out in zip(outputs, grad_outputs):
        if out is not None and out.requires_grad:
            new_outputs += (out,)
            new_grad_outputs += (grad_out,)
    if len(new_outputs) == 0:
        return (None,) * len(inputs)
    else:
        return msadapter.autograd.grad(new_outputs, inputs, new_grad_outputs,
            allow_unused=True, create_graph=create_graph, retain_graph=retain_graph, is_grads_batched=is_grads_batched)

def _fill_in_zeros(grads, refs, strict, create_graph, stage):
    if stage not in ["back", "back_trick", "double_back", "double_back_trick"]:
        raise RuntimeError(f"Invalid stage argument '{stage}' to _fill_in_zeros")
    res: tuple[msadapter.Tensor, ...] = ()
    for i, grads_i in enumerate(grads):
        if grads_i is None:
            if strict:
                if stage == "back":
                    raise RuntimeError("The output of the user-provided function is independent of input {i}. This is not allowed in strict mode.")
                elif stage == "back_trick":
                    raise RuntimeError(
                        f"The gradient with respect to the input is independent of entry {i}"
                        " in the grad_outputs when using the double backward trick to compute"
                        " forward mode gradients. This is not allowed in strict mode.")
                elif stage == "double_back":
                    raise RuntimeError("The jacobian of the user-provided function is independent of input {i}. This is not allowed in strict mode.")
                else:
                    raise RuntimeError(
                        "The hessian of the user-provided function is independent of "
                        f"entry {i} in the grad_jacobian. This is not allowed in strict "
                        "mode as it prevents from using the double backward trick to "
                        "replace forward mode AD.")
            grads_i = msadapter.zeros_like(refs[i])
        else:
            if strict and create_graph and not grads_i.requires_grad:
                if "double" not in stage:
                    raise RuntimeError("The jacobian of the user-provided function is independent of input {i}. This is not allowed in strict mode when create_graph=True.")
                else:
                    raise RuntimeError("The hessian of the user-provided function is independent of input {i}. This is not allowed in strict mode when create_graph=True.")
        res += (grads_i,)
    return res

def vjp(func, inputs, v=None, create_graph=False, strict=False):
    with msadapter.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "vjp")
        _check_requires_grad(outputs, "outputs", strict=strict)
        if v is not None:
            _, v = _as_tuple(v, "v", "vjp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, outputs, is_outputs_tuple)
        else:
            if len(outputs) != 1 or outputs[0].nelement() != 1:
                raise RuntimeError("The vector v can only be None if the user-provided function returns a single Tensor with a single element.")
    enable_grad = True if create_graph else msadapter.is_grad_enabled()
    with msadapter.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)
        vjp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "back")
    outputs = _grad_postprocess(outputs, create_graph)
    vjp = _grad_postprocess(vjp, create_graph)
    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(vjp, is_inputs_tuple)

def jvp(func, inputs, v=None, create_graph=False, strict=False):
    with msadapter.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)
        if v is not None:
            _, v = _as_tuple(v, "v", "jvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError("The vector v can only be None if the input to the user-provided function is a single Tensor with a single element.")
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jvp")
        _check_requires_grad(outputs, "outputs", strict=strict)
        grad_outputs = tuple(msadapter.zeros_like(out, requires_grad=True) for out in outputs)
        grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
        _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)
    if create_graph:
        with msadapter.enable_grad():
            grad_res = _autograd_grad(grad_inputs, grad_outputs, v, create_graph=create_graph)
            jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")
    else:
        grad_res = _autograd_grad(grad_inputs, grad_outputs, v, create_graph=create_graph)
        jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

    outputs = _grad_postprocess(outputs, create_graph)
    jvp = _grad_postprocess(jvp, create_graph)
    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(jvp, is_outputs_tuple)

def _construct_standard_basis_for(
    tensors: tuple[msadapter.Tensor, ...], tensor_numels: tuple[int, ...]
) -> tuple[msadapter.Tensor, ...]:
    assert len(tensors) == len(tensor_numels)
    assert len(tensors) > 0
    total_numel = sum(tensor_numels)
    chunks = tuple(tensor.new_zeros(total_numel, tensor_numel) for tensor, tensor_numel in zip(tensors, tensor_numels))
    diag_start_idx = 0
    for chunk, numel in zip(chunks, tensor_numels):
        chunk.diagonal(diag_start_idx).fill_(1)
        diag_start_idx -= numel
    return chunks

def vhp(func, inputs, v=None, create_graph=False, strict=False):
    with msadapter.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vhp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)
        if v is not None:
            _, v = _as_tuple(v, "v", "vhp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError("The vector v can only be None if the input to the user-provided function is a single Tensor with a single element.")
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "vhp")
        _check_requires_grad(outputs, "outputs", strict=strict)
        if is_outputs_tuple or not isinstance(outputs[0], msadapter.Tensor):
            raise RuntimeError("The function given to vhp should return a single Tensor")
        if outputs[0].nelement() != 1:
            raise RuntimeError("The Tensor returned by the function given to vhp should contain a single element")
        jac = _autograd_grad(outputs, inputs, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)
    enable_grad = True if create_graph else msadapter.is_grad_enabled()
    with msadapter.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(jac, inputs, v, create_graph=create_graph)
        vhp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "double_back")
    outputs = _grad_postprocess(outputs, create_graph)
    vhp = _grad_postprocess(vhp, create_graph)
    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(vhp, is_inputs_tuple)

def hvp(func, inputs, v=None, create_graph=False, strict=False):
    with msadapter.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hvp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)
        if v is not None:
            _, v = _as_tuple(v, "v", "hvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError("The vector v can only be None if the input to the user-provided function is a single Tensor with a single element.")
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "hvp")
        _check_requires_grad(outputs, "outputs", strict=strict)
        if is_outputs_tuple or not isinstance(outputs[0], msadapter.Tensor):
            raise RuntimeError("The function given to hvp should return a single Tensor")
        if outputs[0].nelement() != 1:
            raise RuntimeError("The Tensor returned by the function given to hvp should contain a single element")
        jac = _autograd_grad(outputs, inputs, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)
        grad_jac = tuple(msadapter.zeros_like(inp, requires_grad=True) for inp in inputs)
        double_back = _autograd_grad(jac, inputs, grad_jac, create_graph=True)
        _check_requires_grad(jac, "hessian", strict=strict)
    enable_grad = True if create_graph else msadapter.is_grad_enabled()
    with msadapter.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=create_graph)
        hvp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "double_back_trick")
    outputs = _grad_postprocess(outputs, create_graph)
    hvp = _grad_postprocess(hvp, create_graph)
    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(hvp, is_inputs_tuple)
