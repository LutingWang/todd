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
        priorities = list(priorities)
        queue = list(queue)
        super().__init__(zip(priorities, queue))

    @property
    def priorities(self) -> list[Mapping[KT, int]]:
        return [p for p, _ in self]

    @property
    def queue(self) -> list[VT]:
        return [q for _, q in self]

    def __call__(self, key: KT) -> list[VT]:
        if len(self) == 0:
            return []
        priorities = [p.get(key, 0) for p in self.priorities]
        priority_index = [(p, i) for i, p in enumerate(priorities)]
        priority_index = sorted(priority_index)
        _, indices = zip(*priority_index)
        queue = [self.queue[i] for i in indices]
        return queue
