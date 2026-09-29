__all__ = [
    'PriorityQueue',
]

from collections import UserList
from typing import Iterable, Mapping, TypeVar

K = TypeVar('K')
V = TypeVar('V')


class PriorityQueue(UserList[tuple[Mapping[K, int], V]]):

    def __init__(
        self,
        priorities: Iterable[Mapping[K, int]],
        queue: Iterable[V],
    ) -> None:
        super().__init__(zip(priorities, queue))

    @property
    def priorities(self) -> list[Mapping[K, int]]:
        return [p for p, _ in self]

    @property
    def queue(self) -> list[V]:
        return [q for _, q in self]

    def __call__(self, key: K) -> list[V]:
        return [q for _, q in sorted(self, key=lambda x: x[0].get(key, 0))]
