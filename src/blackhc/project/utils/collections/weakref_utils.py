import weakref
from typing import Dict, Iterator, TypeVar, Generic
from typing import MutableMapping, MutableSet
import objproxies

KT = TypeVar("KT")  # Key type.
VT = TypeVar("VT")  # Value type.
KT_co = TypeVar("KT_co", covariant=True)  # Value type covariant containers.
VT_co = TypeVar("VT_co", covariant=True)  # Value type covariant containers.

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)  # Any type covariant containers.


def supports_weakrefs(value):
    """Determine if the given value supports weak references."""
    return type(value).__weakrefoffset__ != 0


class ObjectProxy(objproxies.ObjectProxy):
    """A proxy object that supports weak references."""

    __slots__ = ("__weakref__",)


class IdMapFinalizer(Generic[KT]):
    """A class that manages finalizers for objects identified by their IDs.

    This class provides methods to register objects with finalizers, check if an object is in the finalizer map,
    get the number of objects, iterate over the objects, and release objects from the finalizer map.
    """

    id_to_finalizer: Dict[int, weakref.finalize]

    def __init__(self):
        """Initialize the IdMapFinalizer with an empty dictionary."""
        self.id_to_finalizer = {}

    def _finalizer(self, id_value, custom_handler):
        """Internal finalizer method to clean up and call a custom handler if provided."""
        del self.id_to_finalizer[id_value]
        if custom_handler is not None:
            custom_handler(id_value)

    def __contains__(self, item):
        """Check if an item is in the finalizer map."""
        return id(item) in self.id_to_finalizer

    def __len__(self):
        """Return the number of items in the finalizer map."""
        return len(self.id_to_finalizer)

    def __iter__(self):
        """Iterate over the objects in the finalizer map."""
        # TODO: add a test that shows that this is necessary to avoid deletions
        # Take a snapshot of the keys. This will ensure that the dictionary will be stable during iteration.
        return iter(list(map(self.get_object, self.id_to_finalizer)))

    def get_object(self, id_value):
        """Get the object associated with the given ID."""
        return self.id_to_finalizer[id_value].peek()[0]

    def register(self, value: KT, custom_handler):
        """Register a value with an optional custom handler for finalization."""
        if not supports_weakrefs(value):
            # TODO: log?
            return

        id_value = id(value)
        if id_value in self.id_to_finalizer:
            raise ValueError(f"{value} has already been added to the finalizer!")

        self.id_to_finalizer[id_value] = weakref.finalize(value, self._finalizer, id_value, custom_handler)

    def release(self, value: KT):
        """Release a value from the finalizer map."""
        id_value = id(value)
        finalizer = self.id_to_finalizer.get(id_value)
        if finalizer is not None:
            finalizer.detach()
            del self.id_to_finalizer[id_value]

    def clear(self):
        """Clear all finalizers from the map."""
        for finalizer in self.id_to_finalizer.values():
            finalizer.detach()
        self.id_to_finalizer.clear()

    def __del__(self):
        """Ensure all finalizers are cleared upon deletion."""
        self.clear()


class WeakIdSet(MutableSet[T]):
    """
    A set that holds weak references to its items.

    This set allows its items to be garbage collected when there are no strong references to them,
    while still maintaining the set membership until the items are collected.
    """

    id_map_finalizer: IdMapFinalizer

    def __init__(self):
        """Initialize the WeakIdSet with an IdMapFinalizer."""
        self.id_map_finalizer = IdMapFinalizer()

    def add(self, x: T) -> None:
        """Add an item to the set."""
        assert supports_weakrefs(x)
        if x not in self.id_map_finalizer:
            self.id_map_finalizer.register(x, None)

    def discard(self, x: T) -> None:
        """Remove an item from the set."""
        self.id_map_finalizer.release(x)

    def __contains__(self, x: object) -> bool:
        """Check if an item is in the set."""
        return x in self.id_map_finalizer

    def __len__(self) -> int:
        """Return the number of items in the set."""
        return len(self.id_map_finalizer)

    def __iter__(self) -> Iterator[T]:
        """Iterate over the items in the set."""
        return iter(self.id_map_finalizer)

    def __repr__(self):
        """Return a string representation of the WeakIdSet."""
        return f"WeakIdSet{{{ ', '.join(map(repr, self))}}}"


class WeakKeyIdMap(MutableMapping[KT, VT]):
    """
    A dictionary-like object that uses weak references for its keys.

    This class allows the keys to be garbage collected when there are no strong references to them,
    while still maintaining the values associated with those keys.
    """

    id_map_to_value: Dict[int, VT]
    id_map_finalizer: IdMapFinalizer

    def __init__(self):
        """Initialize the WeakKeyIdMap with an empty id_map_to_value and an IdMapFinalizer."""
        self.id_map_to_value = {}
        self.id_map_finalizer = IdMapFinalizer()

    def _release(self, id_value):
        """Release the value associated with the given id_value from the map."""
        del self.id_map_to_value[id_value]

    def __setitem__(self, k: KT, v: VT) -> None:
        """Set the value v for the key k in the map."""
        assert supports_weakrefs(k)

        id_k = id(k)
        if id_k not in self.id_map_to_value:
            self.id_map_finalizer.register(k, self._release)
        self.id_map_to_value[id_k] = v

    def __delitem__(self, k: KT) -> None:
        """Delete the item with the key k from the map."""
        del self.id_map_to_value[id(k)]
        self.id_map_finalizer.release(k)

    def __getitem__(self, k: KT) -> VT:
        """Get the value associated with the key k."""
        return self.id_map_to_value[id(k)]

    def __len__(self) -> int:
        """Return the number of items in the map."""
        return len(self.id_map_to_value)

    def __iter__(self) -> Iterator[KT]:
        """Return an iterator over the keys of the map."""
        # TODO: add a test that shows that this is necessary to avoid deletions
        # Take a snapshot of the keys. This will ensure that the dictionary will be stable during iteration.
        return iter(self.id_map_finalizer)

    def __repr__(self):
        """Return a string representation of the WeakKeyIdMap."""
        return f"KeyIdDict{{{', '.join(map(lambda key: f'{repr(key)}:{repr(self[key])}', self))}}}"


class AbstractWrappedValueMutableMapping(Generic[KT, VT, T], MutableMapping[KT, VT]):
    """
    A mutable mapping that wraps values of type VT into another type T for storage.

    This abstract class requires the implementation of methods to convert values to and from the storage type.
    """

    data: Dict[KT, T]

    def __init__(self):
        """Initialize the AbstractWrappedValueMutableMapping with an empty data dictionary."""
        self.data = {}

    def value_to_store(self, v: VT) -> T:
        """Convert the value v to the type T for storage."""
        raise NotImplementedError()

    def store_to_value(self, v: T) -> VT:
        """Convert the stored value v of type T back to the original type VT."""
        raise NotImplementedError()

    def __setitem__(self, k: KT, v: VT) -> None:
        """Set the value v for the key k in the data dictionary."""
        self.data[k] = self.value_to_store(v)

    def __delitem__(self, v: KT) -> None:
        """Delete the item with the key v from the data dictionary."""
        del self.data[v]

    def __getitem__(self, k: KT) -> VT:
        """Get the value associated with the key k from the data dictionary."""
        return self.store_to_value(self.data[k])

    def __len__(self) -> int:
        """Return the number of items in the data dictionary."""
        return len(self.data)

    def __iter__(self) -> Iterator[KT]:
        """Return an iterator over the keys of the data dictionary."""
        return iter(self.data)
