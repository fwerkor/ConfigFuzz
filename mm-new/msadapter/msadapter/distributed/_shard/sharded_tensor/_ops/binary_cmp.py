# mypy: allow-untyped-defs
import msadapter
import msadapter.distributed as dist
import msadapter.distributed.distributed_c10d as distributed_c10d
from msadapter.distributed._shard.sharded_tensor import _sharded_op_impl, ShardedTensor


def _communicate_result(result, pg):
    # Gather results from all ranks.
    if result:
        result_tensor = msadapter.ones(1, device=msadapter.device(msadapter.cuda.current_device()))
    else:
        result_tensor = msadapter.zeros(1, device=msadapter.device(msadapter.cuda.current_device()))

    dist.all_reduce(result_tensor, group=pg)

    expected_result = msadapter.ones(
        1, device=msadapter.device(msadapter.cuda.current_device())
    ) * dist.get_world_size(pg)

    return msadapter.equal(result_tensor, expected_result)


def binary_cmp(cmp_fun, types, args, kwargs=None, process_group=None):
    if len(args) != 2:
        raise ValueError(f"Expected two arguments for msadapter.{cmp_fun.__name__}")

    st1 = args[0]
    st2 = args[1]
    if not (isinstance(st1, ShardedTensor) and isinstance(st2, ShardedTensor)):
        raise TypeError(
            f"Both arguments to msadapter.{cmp_fun.__name__} need to be of type ShardedTensor"
        )

    # Verify same PG
    if st1._process_group != st2._process_group:
        return False

    if distributed_c10d._rank_not_in_group(
        st1._process_group
    ) or distributed_c10d._rank_not_in_group(st2._process_group):
        return distributed_c10d._rank_not_in_group(
            st1._process_group
        ) == distributed_c10d._rank_not_in_group(st2._process_group)

    # Verify metadata
    if st1.metadata() != st2.metadata():
        return _communicate_result(False, st1._process_group)

    # Verify number of local shards
    st1_local_shards = st1.local_shards()
    st2_local_shards = st2.local_shards()
    if len(st1_local_shards) != len(st2_local_shards):
        return _communicate_result(False, st1._process_group)

    # kwargs must be dict-like
    if kwargs is None:
        kwargs = {}
    # Verify each local shard
    for idx in range(len(st1_local_shards)):
        if st1_local_shards[idx].metadata != st2_local_shards[idx].metadata:
            return _communicate_result(False, st1._process_group)
        if not cmp_fun(
            st1_local_shards[idx].tensor, st2_local_shards[idx].tensor, **kwargs
        ):
            return _communicate_result(False, st1._process_group)

    return _communicate_result(True, st1._process_group)


@_sharded_op_impl(msadapter.equal)
def equal(types, args, kwargs, process_group):
    return binary_cmp(msadapter.equal, types, args, kwargs, process_group)


@_sharded_op_impl(msadapter.allclose)
def allclose(types, args, kwargs, process_group):
    return binary_cmp(msadapter.allclose, types, args, kwargs, process_group)
