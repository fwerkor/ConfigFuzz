# Copyright (c) 2024, Huawei Technologies.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from torch_npu.utils.utils import _print_error_log
from mindspore.profiler import mstx

__all__ = ["mstx"]


def _no_exception_func(default_ret=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as ex:
                _print_error_log(f"Call {func.__name__} failed. Exception: {str(ex)}")
                return default_ret
            return result
        return wrapper
    return decorator


class mstx:
    @staticmethod
    @_no_exception_func()
    def mark(message=""):
        return mstx.mark(message)

    @staticmethod
    @_no_exception_func()
    def range_start(message: str, stream=None) -> int:
        return mstx.range_start(message, stream)

    @staticmethod
    @_no_exception_func()
    def range_end(range_id: int):
        return mstx.range_end(range_id)

    @staticmethod
    @_no_exception_func()
    def mstx_range(message: str, stream=None):
        def wrapper(func):
            def inner(*args, **kargs):
                range_id = mstx.range_start(message, stream)
                ret = func(*args, **kargs)
                mstx.range_end(range_id)
                return ret
            return inner
        return wrapper
