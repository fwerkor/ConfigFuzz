import os
import argparse
import unittest

import importlib
import importlib.metadata
from packaging import version
from packaging.version import Version
from datetime import timedelta

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

def init_process_group(
        host: str, port: int, rank: int, world_size: int,
        backend: str) -> ProcessGroup:
    init_method = f"tcp://{host}:{port}"
    backend = Backend(backend)  # it is basically string
    timeout = timedelta(seconds=100)

    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)

    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)

    pg = ProcessGroup(prefix_store, group_rank, group_size)

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

def destroy_process_group(pg: ProcessGroup) -> None:
    """
    Destroy ProcessGroup returned by init_process_group().
    """
    if is_torch_equal_or_newer("2.7"):
        pg.shutdown()
    else:
        # Lazy import for non-CUDA backends.
        torch.distributed.distributed_c10d._shutdown_backend(pg)

    _unregister_process_group(pg.group_name)

def test_allreduce_max1(group, rank, world_size):
    # dtype bool is not supported
    tensor = torch.tensor([True], dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=group)
    assert tensor

def test_allreduce_max2(group, rank, world_size):
    tensor = torch.tensor([False], dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=group)
    assert not tensor

def test_allreduce_max3(group, rank, world_size):
    has_unfinished = True if rank == 0 else False
    tensor = torch.tensor([has_unfinished], dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=group)
    assert tensor.dtype == torch.int32 and tensor

def test_allreduce_max4(group, rank, world_size):
    tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=group)
    expected = torch.tensor([world_size, world_size+1, world_size+2])
    assert tensor.dtype == torch.int32 and tensor.equal(expected)

def test_allreduce_max5(group, rank, world_size):
    tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.int64)
    torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=group)
    expected = torch.tensor([world_size, world_size+1, world_size+2])
    assert tensor.dtype == torch.int64 and tensor.equal(expected)

def test_allreduce_max6(group, rank, world_size):
    tensor = torch.tensor([rank + 1.0])
    torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=group)
    expected = torch.tensor([world_size])
    assert tensor.equal(expected)

def test_allreduce_min1(group, rank, world_size):
    tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=group)
    expected = torch.tensor([1, 2, 3])
    assert tensor.dtype == torch.int32 and tensor.equal(expected)

def test_allreduce_min2(group, rank, world_size):
    tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.int64)
    torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=group)
    expected = torch.tensor([1, 2, 3])
    assert tensor.equal(expected)

def test_allreduce_min3(group, rank, world_size):
    tensor = torch.tensor([rank + 1.0])
    torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=group)
    assert tensor.equal(torch.tensor([1]))

def test_allreduce_sum1(group, rank, world_size):
    tensor = torch.tensor([rank + 1.0])
    torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=group)
    expected = torch.Tensor([torch.arange(1, world_size+1).sum()])
    assert tensor.equal(expected)

def test_allreduce_sum2(group, rank, world_size):
    tensor = torch.tensor(rank + 1.0)
    torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=group)
    expected = torch.Tensor(torch.arange(1, world_size+1).sum())
    assert tensor.equal(expected)

def test_allreduce_sum3(group, rank, world_size):
    tensor = torch.tensor([rank, rank + 1.0])
    ret = torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=group)
    expected = torch.Tensor([torch.arange(0, world_size).sum(), torch.arange(1, world_size+1).sum()])
    assert ret is None
    assert tensor.equal(expected)

def test_allreduce_prod1(group, rank, world_size):
    tensor = torch.tensor([rank + 1.0])
    torch.distributed.all_reduce(tensor, op=ReduceOp.PRODUCT, group=group)
    expected = torch.Tensor([torch.arange(1, world_size+1).prod()])
    assert tensor.equal(expected)

def worker(rank, world_size):
    group = init_process_group("127.0.0.1", 1232, rank, world_size, 'gloo')

    test_allreduce_max1(group, rank, world_size)
    test_allreduce_max2(group, rank, world_size)
    test_allreduce_max3(group, rank, world_size)
    test_allreduce_max4(group, rank, world_size)
    test_allreduce_max5(group, rank, world_size)
    test_allreduce_max6(group, rank, world_size)
    test_allreduce_min1(group, rank, world_size)
    test_allreduce_min2(group, rank, world_size)
    test_allreduce_min3(group, rank, world_size)
    test_allreduce_sum1(group, rank, world_size)
    test_allreduce_sum2(group, rank, world_size)
    test_allreduce_sum3(group, rank, world_size)
    test_allreduce_prod1(group, rank, world_size)

    destroy_process_group(group)


class TestProcessGroupGloo(unittest.TestCase):
    def test_api_completeness(self):
        world_size = 2

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
