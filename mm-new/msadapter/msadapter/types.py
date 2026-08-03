from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
from typing import Any, List, Tuple, IO, TYPE_CHECKING, Union, Dict, Sequence
from typing_extensions import Self, TypeAlias
from mindspore._c_expression.typing import Type as dtype
import msadapter

_TensorOrTensors = Union[msadapter.Tensor, Sequence[msadapter.Tensor]]
Device: TypeAlias = Union[str, int, None]

class device:
    def __init__(self, name, index = None):
        if isinstance(name, device):
            if hasattr(name, "type"):
                self.type = name.type

            if hasattr(name, "index"):
                self.index = name.index

            return

        if index is not None:
            self.type = name
            self.index = index
            return

        names = name.split(":")
        if len(names) == 1:
            self.type = names[0]
            self.index = None
        elif len(names) == 2:
            self.type = names[0]
            self.index = int(names[1])
        else:
            raise ValueError("Invalid device arguments: %s!" % str(name))

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type

    def __repr__(self):
        if self.index is not None:
            return f"device(type='{self.type}', index='{self.index}')"
        return f"device(type='{self.type}')"

    def __enter__(self): ...

    def __exit__(self, type, value, traceback): ...



# Meta-type for "numeric" things; matches our docs
Number: TypeAlias = Union[int, float, bool]
# tuple for isinstance(x, Number) checks.
# FIXME: refactor once python 3.9 support is dropped.
_Number = (int, float, bool)

# Storage protocol implemented by ${Type}StorageBase classes
class Storage:
    _cdata: int
    device: device
    dtype: dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Storage":
        raise NotImplementedError

    def _new_shared(self, size: int) -> "Storage":
        raise NotImplementedError

    def _write_file(
        self,
        f: Any,
        is_real_file: bool,
        save_size: bool,
        element_size: int,
    ) -> None:
        raise NotImplementedError

    def element_size(self) -> int:
        raise NotImplementedError

    def is_shared(self) -> bool:
        raise NotImplementedError

    def share_memory_(self) -> "Storage":
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError

    def cpu(self) -> "Storage":
        raise NotImplementedError

    def data_ptr(self) -> int:
        raise NotImplementedError

    def from_file(
        self,
        filename: str,
        shared: bool = False,
        nbytes: int = 0,
    ) -> "Storage":
        raise NotImplementedError

    def _new_with_file(
        self,
        f: Any,
        element_size: int,
    ) -> "Storage":
        raise NotImplementedError

_device = device
_dtype = dtype
_size = Union[msadapter.Size, List[_int], Tuple[_int, ...]]


class UntypedStorage:
    """
    UntypedStorage implementation for safetensors compatibility.

    This class provides a PyTorch-compatible UntypedStorage API.
    It can be called as a function (UntypedStorage(size)) or used with
    class methods (UntypedStorage.from_file(...)).
    """

    def __init__(self, size=0, *, device=None):
        """
        Create an UntypedStorage with given size in bytes.

        Args:
            size: Size in bytes
            device: Device to create storage on (ignored for compatibility)
        """
        self._size = size
        self._data = bytearray(size)
        self._device = device

    @classmethod
    def from_file(cls, filename, shared, nbytes):
        """Load storage from file (for torch.load and safetensors compatibility)."""
        return cls._from_file(filename, shared, nbytes)

    @classmethod
    def _from_file(cls, filename, shared, nbytes):
        """Load storage from file (for torch.load compatibility)."""
        instance = cls.__new__(cls)
        with open(filename, 'rb') as f:
            data = f.read(nbytes) if nbytes > 0 else f.read()
            instance._data = bytearray(data)
            instance._size = len(instance._data)
        instance._device = None
        return instance

    @classmethod
    def _from_buffer(cls, buffer, *, byte_order=None):
        """Create storage from a buffer (bytes-like object)."""
        instance = cls.__new__(cls)
        instance._data = bytearray(buffer)
        instance._size = len(buffer)
        instance._device = None
        return instance

    def __getitem__(self, idx):
        """Get byte at index."""
        return self._data[idx]

    def __setitem__(self, idx, value):
        """Set byte at index."""
        self._data[idx] = value

    def __len__(self):
        """Return size in bytes."""
        return self._size

    def size(self):
        """Return size in bytes."""
        return self._size

    def nbytes(self):
        """Return size in bytes."""
        return self._size

    @property
    def _untyped(self):
        """Return self for safetensors compatibility."""
        return self

    @property
    def untyped(self):
        """Return self (for PyTorch API compatibility)."""
        return self

    def data_ptr(self):
        """Return pointer to data (for compatibility)."""
        return id(self._data)

    def cpu(self):
        """Return self (for device compatibility)."""
        return self

    def to(self, device):
        """Move to device (for compatibility, currently no-op)."""
        self._device = device
        return self

    def clone(self):
        """Create a copy of this storage."""
        instance = self.__class__.__new__(self.__class__)
        instance._data = bytearray(self._data)
        instance._size = self._size
        instance._device = self._device
        return instance

    def copy_(self, other):
        """Copy data from another storage."""
        if isinstance(other, UntypedStorage):
            self._data = bytearray(other._data)
            self._size = other._size
        else:
            self._data = bytearray(other)
            self._size = len(self._data)
        return self

    def resize_(self, size):
        """Resize storage to new size."""
        if size > self._size:
            self._data.extend(b'\x00' * (size - self._size))
        elif size < self._size:
            self._data = self._data[:size]
        self._size = size
        return self

    def byte(self):
        """Return self as byte storage (for dtype compatibility)."""
        return self

    def byteswap(self, dtype):
        """Byte swap (for endianness compatibility, currently no-op)."""
        pass

    def element_size(self):
        """Return element size in bytes (always 1 for UntypedStorage)."""
        return 1

    def is_shared(self):
        """Check if storage is shared (always False)."""
        return False

    def share_memory_(self):
        """Share memory (no-op for compatibility)."""
        return self

    def _write_file(self, f, is_real_file, save_size, element_size):
        """Write storage to file (for torch.save compatibility)."""
        import struct
        if save_size:
            f.write(struct.pack('<Q', self._size // element_size))
        f.write(self._data)

    def _new_shared(self, size):
        """Create new shared storage (for compatibility)."""
        return UntypedStorage(size)

    def _new_with_file(self, f, element_size):
        """Create new storage from file (for torch.load compatibility)."""
        return UntypedStorage._from_file(f.name, False, 0)

    def __deepcopy__(self, memo):
        """Deep copy support."""
        return self.clone()

    def tolist(self):
        """Convert to list of bytes."""
        return list(self._data)

    def numpy(self):
        """Convert to numpy array (view)."""
        import numpy as np
        return np.frombuffer(self._data, dtype=np.uint8)