import msadapter

from .functional import *  # noqa: F403


if msadapter.distributed.rpc.is_available():
    from .api.remote_module import RemoteModule
