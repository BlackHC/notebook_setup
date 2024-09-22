from typing import MutableSet, TypeVar, Iterator, Dict

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)  # Any type covariant containers.


# TODO: add tests
class IdSet(MutableSet[T]):
    id_value: Dict[int, T]

    def __init__(self):
        self.id_value = {}

    def add(self, x: T) -> None:
        self.id_value[id(x)] = x

    def discard(self, x: T) -> None:
        if x in self:
            del self.id_value[id(x)]

    def clear(self) -> None:
        self.id_value.clear()

    def __contains__(self, x: object) -> bool:
        return id(x) in self.id_value

    def __len__(self) -> int:
        return len(self.id_value)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.id_value.values())

    def __repr__(self):
        return f"IdSet{{{ ', '.join(map(repr, self))}}}"


