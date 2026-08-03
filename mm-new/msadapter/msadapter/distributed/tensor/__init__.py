# Copyright (c) Meta Platforms, Inc. and affiliates

import msadapter
# import msadapter.distributed.tensor._ops  # force import all built-in dtensor ops
from msadapter.distributed.device_mesh import DeviceMesh, init_device_mesh  # noqa: F401
from msadapter.distributed.tensor._api import (
    distribute_module,
    distribute_tensor,
    DTensor,
    empty,
    full,
    ones,
    rand,
    randn,
    zeros,
)
from msadapter.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from msadapter.optim.optimizer import (
    _foreach_supported_types as _optim_foreach_supported_types,
)
# from msadapter.utils._foreach_utils import (
#     _foreach_supported_types as _util_foreach_supported_types,
# )


# All public APIs from dtensor package
__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "Shard",
    "Replicate",
    "Partial",
    "Placement",
    "ones",
    "empty",
    "full",
    "rand",
    "randn",
    "zeros",
]

# For weights_only msadapter.load
from ._dtensor_spec import DTensorSpec as _DTensorSpec, TensorMeta as _TensorMeta


# msadapter.serialization.add_safe_globals(
#     [
#         DeviceMesh,
#         _DTensorSpec,
#         _TensorMeta,
#         DTensor,
#         Partial,
#         Replicate,
#         Shard,
#     ]
# )


# Append DTensor to the list of supported types for foreach implementation for optimizer
# and clip_grad_norm_ so that we will try to use foreach over the for-loop implementation on CUDA.
if DTensor not in _optim_foreach_supported_types:
    _optim_foreach_supported_types.append(DTensor)

# if DTensor not in _util_foreach_supported_types:
#     _util_foreach_supported_types.append(DTensor)


# Set namespace for exposed private names
DTensor.__module__ = "msadapter.distributed.tensor"
distribute_tensor.__module__ = "msadapter.distributed.tensor"
distribute_module.__module__ = "msadapter.distributed.tensor"
ones.__module__ = "msadapter.distributed.tensor"
empty.__module__ = "msadapter.distributed.tensor"
full.__module__ = "msadapter.distributed.tensor"
rand.__module__ = "msadapter.distributed.tensor"
randn.__module__ = "msadapter.distributed.tensor"
zeros.__module__ = "msadapter.distributed.tensor"
