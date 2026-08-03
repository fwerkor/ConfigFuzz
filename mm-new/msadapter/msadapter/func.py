from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import msadapter
import msadapter.nn as nn
from msadapter import Tensor


def functional_call(
    module: "msadapter.nn.Module",
    parameter_and_buffer_dicts: Union[Dict[str, Tensor], Sequence[Dict[str, Tensor]]],
    args: Union[Any, Tuple],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    tie_weights: bool = True,
    strict: bool = False
):
    return module(*args, **kwargs)
