# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Defines communication operators with functional form.
"""
from mindspore.communication import GlobalComm, get_group_size
from mindspore.communication._comm_helper import _get_rank_helper
from mindspore.ops.auto_generate.gen_ops_prim import dist_comm_all_to_all_v_c_op
from mindspore._c_expression import CommHandle as CommHandle_
from mindspore import jit_class
import mindspore as ms

__all__ = [
    'all_to_all_v_c'
]

import mindspore.ops.operations as P

_GROPU_SIZE_CACHE = {}
_GROPU_RANK_CACHE = {}

@jit_class
class CommHandle(CommHandle_):
    r"""
    Usually, handles are created in C++during the execution of communication operators and returned to the Python
    layer. It will not be created directly in Python. Only in scenarios where graph patterns are compatible,
    handles will be created using Python.
    """

    def __init__(self, handle=None, exec_sync=False):
        super(CommHandle, self).__init__()
        self.handle = handle
        self.exec_sync = exec_sync


    def wait(self):
        r"""
        The wait for asynchronous handles will not take effect for handles created on the Python side.

        >>> import numpy as np
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import all_reduce
        >>> from mindspore import Tensor
        >>>
        >>> init()
        >>> input_tensor = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> output, handle = all_reduce(input_tensor, async_op=True)
        >>> handle.wait()
        >>> print(output)
        [[2. 2. 2. 2. 2. 2. 2. 2.]
         [2. 2. 2. 2. 2. 2. 2. 2.]]
        """
        if self.handle:
            self.handle.wait()
        if self.exec_sync:
            ms.runtime.synchronize()


default_handle = CommHandle()


def _get_size(shape):
    numel = 1
    for s in shape:
        numel *= s
    return numel


def _deal_comm_outputs(output, async_op, exec_sync=False):
    """
    deal with comm ops outputs.
    """
    if isinstance(output, tuple):
        if not async_op:
            output[1].wait()
            if exec_sync:
                ms.runtime.synchronize()
            return None
        return CommHandle(output[1], exec_sync)

    if not async_op:
        return None
    return default_handle


def _get_all_to_all_v_c_numel_list(output, input, send_count_matrix_size):
    """get numel list for all_to_all_v_c."""
    send_size_without_first_dim = _get_size(input.shape[1:])
    recv_size_without_first_dim = _get_size(output.shape[1:])
    if send_size_without_first_dim != recv_size_without_first_dim:
        raise ValueError("The input and output dimensions except 0 must be of equal size, "
                         f"but got {send_size_without_first_dim} and {recv_size_without_first_dim}.")
    send_count_matrix = [size * send_size_without_first_dim for size in send_count_matrix_size]
    return send_count_matrix


def get_cache_group_rank(group=GlobalComm.WORLD_COMM_GROUP):
    """get cache rank id."""
    global _GROPU_RANK_CACHE
    if group not in _GROPU_RANK_CACHE:
        _GROPU_RANK_CACHE[group] = _get_rank_helper(group)
    group_rank = _GROPU_RANK_CACHE[group]
    return group_rank


def all_to_all_v_c(output, input, send_count_matrix, group=None, async_op=False):
    r"""
    Based on the user-specified split size, the input tensor is divided and sent to other devices, where split chunks
    are received and then merged into a single output tensor.

    Note:
        Only support PyNative mode, Graph mode is not currently supported.

    Args:
        output (Tensor): the output tensor is gathered concatenated from remote ranks.
        input (Tensor): tensor to be scattered to remote rank.
        send_count_matrix (list[int]): The sending and receiving parameters of all ranks,
            :math:`\text{send_count_matrix}[i*\text{rank_size}+j]` represents the amount of data sent by
            rank i to rank j, and the basic unit is first dimension sizes. Among them, `rank_size`
            indicates the size of the communication group.
        group (str, optional): The communication group to work on. If ``None``, which means ``"hccl_world_group"`` in
            Ascend. Default: ``None``.
        async_op (bool, optional): Whether this operator should be an async operator. Default: ``False`` .

    Returns:
        CommHandle. CommHandle is an async work handle, if `async_op` is set to True.
        CommHandle will be None, when `async_op` is False.

    Raises:
        TypeError: If `input` or `output` is not tensor. `group` is not a str, or async_op is not bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore.mint.distributed import init_process_group, get_rank
        >>> from msadapter.ops.comm_func import all_to_all_v_c
        >>> from mindspore import Tensor
        >>> from mindspore.ops import zeros
        >>>
        >>> init_process_group()
        >>> this_rank = get_rank()
        >>> if this_rank == 0:
        ...     output = Tensor(np.zeros([3]).astype(np.float32))
        ...     tensor = Tensor([0, 1, 2.]) * this_rank
        ...     result = all_to_all_v_c(output, tensor, [0, 3, 3, 0])
        ...     print(output)
        >>> if this_rank == 1:
        ...     output = Tensor(np.zeros([3]).astype(np.float32))
        ...     tensor = Tensor([0, 1, 2.]) * this_rank
        ...     result = all_to_all_v_c(output, tensor, [0, 3, 3, 0])
        ...     print(output)
        rank 0:
        [0. 1. 2]
        rank 1:
        [0. 0. 0]
    """

    if group is None:
        group = GlobalComm.WORLD_COMM_GROUP
    global _GROPU_SIZE_CACHE
    if group not in _GROPU_SIZE_CACHE:
        _GROPU_SIZE_CACHE[group] = get_group_size(group)
    rank_size = _GROPU_SIZE_CACHE[group]
    _send_count_matrix = _get_all_to_all_v_c_numel_list(output, input, send_count_matrix)
    rank_id = get_cache_group_rank(group)
    result = dist_comm_all_to_all_v_c_op(
        output,
        input,
        group,
        _send_count_matrix,
        rank_size,
        rank_id,
    )
    handle = _deal_comm_outputs(result, async_op)
    return handle

