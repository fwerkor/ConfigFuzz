# mypy: allow-untyped-defs
from __future__ import annotations

import collections.abc as _collections_abc
import weakref
from collections.abc import Mapping, MutableMapping
from weakref import ref

from msadapter import Tensor


WeakRef = ref
__all__ = ["TensorWeakRef", "WeakIdRef", "WeakIdKeyDictionary", "WeakTensorKeyDictionary"]


class _IterationGuard:
    def __init__(self, weakcontainer):
        self.weakcontainer = ref(weakcontainer)

    def __enter__(self):
        w = self.weakcontainer()
        if w is not None:
            w._iterating.add(self)
        return self

    def __exit__(self, e, t, b):
        w = self.weakcontainer()
        if w is not None:
            s = w._iterating
            s.remove(self)
            if not s:
                w._commit_removals()

class WeakIdRef(weakref.ref):
    __slots__ = ["_id"]

    def __init__(self, key, callback=None):
        self._id = id(key)
        super().__init__(key, callback)  # type: ignore[call-arg]

    def __call__(self):
        r = super().__call__()
        if hasattr(r, "_fix_weakref"):
            r._fix_weakref()  # type: ignore[union-attr]
        return r

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        a = self()
        b = other()
        if a is not None and b is not None:
            return a is b
        return self is other

class _WeakHashRef(weakref.ref):
    __slots__ = ["_id"]

    def __init__(self, key, callback=None):
        self._id = hash(key)
        super().__init__(key, callback)  # type: ignore[call-arg]

    def __call__(self):
        r = super().__call__()
        if hasattr(r, "_fix_weakref"):
            r._fix_weakref()  # type: ignore[union-attr]
        return r

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        a = self()
        b = other()
        if a is not None and b is not None:
            return hash(a) == hash(b)
        return self is other

class WeakIdKeyDictionary(MutableMapping):
    def __init__(self, dict=None, ref_type=WeakIdRef):  # CHANGED
        self.data = {}
        self.ref_type = ref_type  # CHANGED
        def remove(k, selfref=ref(self)):
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(k)
                else:
                    try:
                        del self.data[k]
                    except KeyError:
                        pass

        self._remove = remove
        self._pending_removals = []
        self._iterating = set()
        self._dirty_len = False
        if dict is not None:
            self.update(dict)

    def _commit_removals(self):
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return
            try:
                del d[key]
            except KeyError:
                pass

    def _scrub_removals(self):
        d = self.data
        self._pending_removals = [k for k in self._pending_removals if k in d]
        self._dirty_len = False

    def __delitem__(self, key):
        self._dirty_len = True
        del self.data[self.ref_type(key)]  # CHANGED

    def __getitem__(self, key):
        return self.data[self.ref_type(key)]  # CHANGED

    def __len__(self):
        if self._dirty_len and self._pending_removals:
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)

    def __repr__(self):
        return f"<{self.__class__.__name__} at {id(self):#x}>"

    def __setitem__(self, key, value):
        self.data[self.ref_type(key, self._remove)] = value  # CHANGED

    def copy(self):
        new = WeakIdKeyDictionary()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = value
        return new

    __copy__ = copy

    def __deepcopy__(self, memo):
        from copy import deepcopy

        new = self.__class__()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = deepcopy(value, memo)
        return new

    def get(self, key, default=None):
        return self.data.get(self.ref_type(key), default)  # CHANGED

    def __contains__(self, key):
        try:
            wr = self.ref_type(key)  # CHANGED
        except TypeError:
            return False
        return wr in self.data

    def items(self):
        with _IterationGuard(self):
            for wr, value in self.data.items():
                key = wr()
                if key is not None:
                    yield key, value

    def keys(self):
        with _IterationGuard(self):
            for wr in self.data:
                obj = wr()
                if obj is not None:
                    yield obj

    __iter__ = keys

    def values(self):
        with _IterationGuard(self):
            for wr, value in self.data.items():
                if wr() is not None:
                    yield value

    def keyrefs(self):
        return list(self.data)

    def popitem(self):
        self._dirty_len = True
        while True:
            key, value = self.data.popitem()
            o = key()
            if o is not None:
                return o, value

    def pop(self, key, *args):
        self._dirty_len = True
        return self.data.pop(self.ref_type(key), *args)  # CHANGED

    def setdefault(self, key, default=None):
        return self.data.setdefault(
            self.ref_type(key, self._remove), default
        )

    def update(self, dict=None, **kwargs):  # type: ignore[override]
        d = self.data
        if dict is not None:
            if not hasattr(dict, "items"):
                dict = type({})(dict)
            for key, value in dict.items():
                d[self.ref_type(key, self._remove)] = value  # CHANGED
        if len(kwargs):
            self.update(kwargs)

    def __ior__(self, other):
        self.update(other)
        return self

    def __or__(self, other):
        if isinstance(other, _collections_abc.Mapping):
            c = self.copy()
            c.update(other)
            return c
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, _collections_abc.Mapping):
            c = self.__class__()
            c.update(other)
            c.update(self)
            return c
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return {id(k): v for k, v in self.items()} == {
            id(k): v for k, v in other.items()
        }

WeakTensorKeyDictionary = WeakIdKeyDictionary

class TensorWeakRef:
    ref: WeakRef[Tensor]
    def __init__(self, tensor: Tensor):
        assert isinstance(tensor, Tensor)
        self.ref = weakref.ref(tensor)

    def __call__(self):
        out = self.ref()
        if out is None:
            return out
        assert isinstance(out, Tensor)
        # TODO, add _fix_weakref type binding
        out._fix_weakref()  # type: ignore[attr-defined]
        return out