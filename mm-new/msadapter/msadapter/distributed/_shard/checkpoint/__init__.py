# Keep old package for BC purposes, this file should be removed once
# everything moves to the `msadapter.distributed.checkpoint` package.
import sys
import warnings

import msadapter
from msadapter.distributed.checkpoint import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`msadapter.distributed._shard.checkpoint` will be deprecated, "
        "use `msadapter.distributed.checkpoint` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules["msadapter.distributed._shard.checkpoint"] = msadapter.distributed.checkpoint
