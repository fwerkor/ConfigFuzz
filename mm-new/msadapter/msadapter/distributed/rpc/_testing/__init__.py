# mypy: allow-untyped-defs

import msadapter


def is_available():
    return hasattr(msadapter._C, "_faulty_agent_init")


if is_available() and not msadapter._C._faulty_agent_init():
    raise RuntimeError("Failed to initialize msadapter.distributed.rpc._testing")

if is_available():
    # Registers FAULTY_TENSORPIPE RPC backend.
    from msadapter._C._distributed_rpc_testing import (
        FaultyTensorPipeAgent,
        FaultyTensorPipeRpcBackendOptions,
    )

    from . import faulty_agent_backend_registry
