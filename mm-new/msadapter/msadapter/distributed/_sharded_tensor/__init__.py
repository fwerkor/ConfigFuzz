# Keep old package for BC purposes, this file should be removed once
# everything moves to the `msadapter.distributed._shard` package.
import sys
import warnings

import msadapter
from msadapter.distributed._shard.sharded_tensor import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`msadapter.distributed._sharded_tensor` will be deprecated, "
        "use `msadapter.distributed._shard.sharded_tensor` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules[
    "msadapter.distributed._sharded_tensor"
] = msadapter.distributed._shard.sharded_tensor
