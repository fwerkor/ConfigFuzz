import msadapter.distributed._shard.sharded_tensor._ops.misc_ops
import msadapter.distributed._shard.sharded_tensor._ops.tensor_ops

# Import all ChunkShardingSpec ops
from msadapter.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding import (
    sharded_embedding,
)
from msadapter.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding_bag import (
    sharded_embedding_bag,
)

from .binary_cmp import allclose, equal
from .init import constant_, kaiming_uniform_, normal_, uniform_
