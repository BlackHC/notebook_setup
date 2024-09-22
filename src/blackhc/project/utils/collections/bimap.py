from dataclasses import dataclass
from typing import MutableMapping, Generic, TypeVar, Optional

from persistent.mapping import PersistentMapping
from persistent import Persistent


KT = TypeVar("KT")  # Key type.
VT = TypeVar("VT")  # Value type.


class Bimap(Generic[KT, VT]):
    def update(self, key: KT, value: Optional[VT]):
        if key is None and value is None:
            raise ValueError("key and value both None!")

        if key is None:
            self.del_value(value)
        elif value is None:
            self.del_key(key)
        else:
            existing_key = self.get_key(value)
            existing_value = self.get_value(key)

            if key is existing_key and value is existing_value:
                return

            if existing_key is not None:
                self._del_key(existing_key)

            if existing_value is not None:
                self._del_value(existing_value)

            self.put_key_value(key, value)

    def put_key_value(self, key: KT, value: VT):
        if key is None or value is None:
            if key is not None:
                raise ValueError("Value is None! Use update instead!")
            if value is not None:
                raise ValueError("Key is None! Use update instead!")
            raise ValueError("Key and value are None! Use update instead!")

        self._put_key_value(key, value)

    def del_key(self, key: KT):
        existing_value = self.get_value(key)
        if existing_value is None:
            return
        self._del_key(key)
        self._del_value(existing_value)

    def del_value(self, value: VT):
        existing_key = self.get_key(value)
        if existing_key is None:
            return
        self._del_key(existing_key)
        self._del_value(value)

    def length(self):
        raise NotImplementedError()

    def get_key(self, value: VT) -> KT:
        raise NotImplementedError()

    def get_value(self, key: KT) -> VT:
        raise NotImplementedError()

    def _del_key(self, key: KT):
        raise NotImplementedError()

    def _del_value(self, value: VT):
        raise NotImplementedError()

    def _put_key_value(self, key: KT, value: VT):
        raise NotImplementedError()

    # TODO: replace this and below by two methods: keys() and values() which both return sets!
    def __contains__(self, key: KT) -> bool:
        raise NotImplementedError()

    def has_value(self, value: VT) -> bool:
        raise NotImplementedError()


@dataclass
class MappingBimap(Bimap[KT, VT]):
    key_value: MutableMapping[KT, VT]
    value_key: MutableMapping[VT, KT]

    def length(self):
        return len(self.key_value)

    def get_key(self, value):
        return self.value_key.get(value)

    def get_value(self, key):
        return self.key_value.get(key)

    def _del_key(self, key):
        del self.key_value[key]

    def _del_value(self, value):
        del self.value_key[value]

    def _put_key_value(self, key, value):
        self.key_value[key] = value
        self.value_key[value] = key

    def __contains__(self, key):
        return key in self.key_value

    def has_value(self, value):
        return value in self.value_key

    def get_keys(self):
        return self.key_value.keys()


class DictBimap(MappingBimap[KT, VT]):
    def __init__(self):
        super().__init__({}, {})


class PersistentBimap(MappingBimap[KT, VT], Persistent):
    def __init__(self):
        super().__init__(PersistentMapping(), PersistentMapping())
