__all__ = [
    'PriorityQueue',
]

from collections import UserList
from typing import Iterable, Mapping, TypeVar

KT = TypeVar('KT')
VT = TypeVar('VT')


class PriorityQueue(UserList[tuple[Mapping[KT, int], VT]]):

    def __init__(
        self,
        priorities: Iterable[Mapping[KT, int]],
        queue: Iterable[VT],
    ) -> None:
        super().__init__(zip(priorities, queue))

    @property
    def priorities(self) -> list[Mapping[KT, int]]:
        return [p for p, _ in self]

    @property
    def queue(self) -> list[VT]:
        return [q for _, q in self]

    def __call__(self, key: KT) -> list[VT]:
        return [q for _, q in sorted(self, key=lambda x: x[0].get(key, 0))]
