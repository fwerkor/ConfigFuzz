import ctypes
from typing import Union
from msadapter.types import Device
import mindspore

class _CUDAAllocator:
    r"""Wrapper over internal CUDA memory allocators."""

    def __init__(self, allocator):
        self._allocator = allocator

    def allocator(self):
        return self._allocator


class CUDAPluggableAllocator(_CUDAAllocator):
    r"""CUDA memory allocator loaded from a so file."""

    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        r"""Memory allocators are compiled in .so files and loaded dynamically using ctypes.

        To change the active allocator use the :func:`msadapter.memory.cuda.change_current_allocator` function.

        Args:
            path_to_so_file(str): Path in the filesystem to the `.so` file containing
                the allocator functions
            alloc_fn_name(str): Name of the function to perform the memory allocation
                in the so file. The signature must be:
                void* alloc_fn_name(ssize_t size, int device, cudaStream_t stream);
            free_fn_name(str): Name of the function to perform the memory release
                in the so file. The signature must be:
                void free_fn_name(void* ptr, size_t size, cudaStream_t stream);

        .. warning::
            This is currently supported only in unix OSs

        .. note::
            See :ref:`cuda-memory-management` for details on creating and using a custom allocator
        """
        allocator = ctypes.CDLL(path_to_so_file)
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None

def max_memory_reserved(device: Union[Device, int] = None) -> int:
    return mindspore.runtime.max_memory_reserved()

def memory_allocated(device: Union[Device, int] = None) -> int:
    return mindspore.runtime.memory_allocated()

def memory_reserved(device: Union[Device, int] = None) -> int:
    return mindspore.runtime.memory_reserved()

def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    return mindspore.runtime.memory_summary()

def reset_max_memory_allocated(device: Union[Device, int] = None) -> None:
    return mindspore.runtime.reset_max_memory_allocated()

def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    return mindspore.runtime.reset_peak_memory_stats()
