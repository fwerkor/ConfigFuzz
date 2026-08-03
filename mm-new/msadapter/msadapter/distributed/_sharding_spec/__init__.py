# Keep old package for BC purposes, this file should be removed once
# everything moves to the `msadapter.distributed._shard` package.
import sys
import warnings

import msadapter
from msadapter.distributed._shard.sharding_spec import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`msadapter.distributed._sharding_spec` will be deprecated, "
        "use `msadapter.distributed._shard.sharding_spec` instead",
        DeprecationWarning,
        stacklevel=2,
    )

import msadapter.distributed._shard.sharding_spec as _sharding_spec


sys.modules["msadapter.distributed._sharding_spec"] = _sharding_spec
