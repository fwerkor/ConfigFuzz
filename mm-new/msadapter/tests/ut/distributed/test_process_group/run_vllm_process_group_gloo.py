import os
import argparse
import unittest

import importlib
import importlib.metadata
from datetime import timedelta
from packaging import version
from packaging.version import Version

import torch
from torch.distributed import ProcessGroup, ReduceOp
from torch.distributed.distributed_c10d import (Backend, PrefixStore,
                                                _get_default_timeout,
                                                _unregister_process_group)
from torch.distributed.rendezvous import rendezvous

def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        torch_version = version.parse(str(torch.__version__))
        return torch_version >= version.parse(target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version('torch')) >= Version(target)

def get_tcp_uri(ip: str, port: int) -> str:
    # Brackets are not permitted in ipv4 addresses,
    # see https://github.com/python/cpython/issues/103848
    return f"tcp://[{ip}]:{port}" if ":" in ip else f"tcp://{ip}:{port}"

def init_gloo_process_group(backend: Backend, prefix_store: PrefixStore,
                            group_rank: int, group_size: int,
                            timeout: timedelta) -> ProcessGroup:
    """
    Stateless init ProcessGroup with gloo backend compatible with 
    different torch versions.
    """
    if is_torch_equal_or_newer("2.6"):
        pg = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )
    else:
        options = ProcessGroup.Options(backend=backend)
        pg = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
            options,
        )
    from torch.distributed.distributed_c10d import ProcessGroupGloo
    backend_class = ProcessGroupGloo(prefix_store,
                                     group_rank,
                                     group_size,
                                     timeout=timeout)
    backend_type = ProcessGroup.BackendType.GLOO
    device = torch.device("cpu")
    if is_torch_equal_or_newer("2.6"):
        # _set_default_backend is supported in torch >= 2.6
        pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()

    pg._register_backend(device, backend_type, backend_class)
    return pg

def stateless_init_torch_distributed_process_group(
        host: str, port: int, rank: int, world_size: int,
        backend: str) -> ProcessGroup:
    """
    A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state. The created ProcessGroup object can be used for
    some operations such as `allreduce`, because it does not depend on the
    global rank. However, some operations such as `broadcast` cannot be used
    because it depends on the global rank.
    """
    init_method = get_tcp_uri(host, port)
    backend = Backend(backend)  # it is basically string
    timeout = _get_default_timeout(backend)

    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)

    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)

    if backend == "gloo":
        return init_gloo_process_group(backend=backend,
                                       prefix_store=prefix_store,
                                       group_rank=group_rank,
                                       group_size=group_size,
                                       timeout=timeout)
    assert False

def stateless_destroy_torch_distributed_process_group(
        pg: ProcessGroup) -> None:
    """
    Destroy ProcessGroup returned by
        stateless_init_torch_distributed_process_group().
    """
    if is_torch_equal_or_newer("2.7"):
        pg.shutdown()
    else:
        # Lazy import for non-CUDA backends.
        from torch.distributed.distributed_c10d import _shutdown_backend
        _shutdown_backend(pg)

    _unregister_process_group(pg.group_name)


class ParallelConfig:
    def __init__(self, host: str, port: int, rank: int, world_size: int):
        self.data_parallel_master_ip = host
        self.get_next_dp_init_port = port
        self.data_parallel_rank = rank
        self.data_parallel_size = world_size

    def stateless_init_dp_group(self) -> "ProcessGroup":
        # use gloo since the engine process might not have cuda device
        dp_group = stateless_init_torch_distributed_process_group(
            self.data_parallel_master_ip,
            self.get_next_dp_init_port,
            self.data_parallel_rank,
            self.data_parallel_size,
            backend="gloo")

        return dp_group

    @staticmethod
    def has_unfinished_dp(dp_group: "ProcessGroup",
                          has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished],
                              dtype=torch.int32,
                              device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished


def worker(rank, world_size):
    config = ParallelConfig("127.0.0.1", 1234, rank, world_size)
    group = config.stateless_init_dp_group()
    has_unfinished = ParallelConfig.has_unfinished_dp(group, False)
    assert not has_unfinished
    stateless_destroy_torch_distributed_process_group(group)


class TestProcessGroupGloo(unittest.TestCase):
    def test_api_completeness(self):
        world_size = 4

        processes = []
        import multiprocessing
        for rank in range(world_size):
            p = multiprocessing.Process(target=worker, args=(rank, world_size))
            processes.append(p)
            p.start()

        exitcode = 0
        for p in processes:
            p.join()
            if p.exitcode != 0:
                exitcode = p.exitcode

        if exitcode != 0:
            exit(exitcode)


if __name__ == "__main__":
    print(f"PYTHONPATH is: {os.getenv('PYTHONPATH')}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance', 'precision', 'outlier'],
                        help="test mode")
    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestProcessGroupGloo().test_api_completeness()
