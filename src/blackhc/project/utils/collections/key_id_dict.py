from typing import TypeVar, MutableMapping, Dict, Tuple, Iterator

KT = TypeVar("KT")  # Key type.
VT = TypeVar("VT")  # Value type.
KT_co = TypeVar('KT_co', covariant=True)  # Value type covariant containers.
VT_co = TypeVar('VT_co', covariant=True)  # Value type covariant containers.


class KeyIdDict(MutableMapping[KT, VT]):
    id_key_value: Dict[int, Tuple[KT, VT]]

    def __init__(self):
        self.id_key_value = {}

    def __setitem__(self, k: KT, v: VT) -> None:
        self.id_key_value[id(k)] = (k, v)

    def __delitem__(self, k: KT) -> None:
        del self.id_key_value[id(k)]

    def __getitem__(self, k: KT) -> VT:
        return self.id_key_value[id(k)][1]

    def __len__(self) -> int:
        return len(self.id_key_value)

    def __iter__(self) -> Iterator[KT]:
        for key, value in self.id_key_value.values():
            yield key

    def __repr__(self):
        return f"KeyIdDict{{{', '.join(map(lambda key_value: f'{repr(key_value[0])}:{repr(key_value[1])}', self.id_key_value.values()))}}}"
