from typing import Any, Callable, Optional, overload, TYPE_CHECKING, TypeVar, Union


def mark_static_address(t: Any, guard: bool = True) -> None:
    """
    Marks an input tensor whose data_ptr will not change across multiple calls
    to a dynamo-compiled function. This indicates to cudagraphs that an extra allocation
    is not needed for this input. The data_ptr will be guarded if guard=True. Note:
    Tensors marked in this way will be kept alive until `msadapter._dynamo.reset()` is called.
    """

    pass

def allow_in_graph(fn):
    """
    Customize which functions TorchDynamo will include in the generated
    graph. Similar to `msadapter.fx.wrap()`.
    ::

        msadapter._dynamo.allow_in_graph(my_custom_function)

        @msadapter._dynamo.optimize(...)
        def fn(a):
            x = msadapter.add(x, 1)
            x = my_custom_function(x)
            x = msadapter.add(x, 1)
            return x

        fn(...)

    Will capture a single graph containing `my_custom_function()`.
    """
    if isinstance(fn, (list, tuple)):
        return [allow_in_graph(x) for x in fn]
    assert callable(fn), "allow_in_graph expects a callable"
    return fn