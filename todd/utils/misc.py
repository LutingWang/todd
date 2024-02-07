__all__ = [
    'get_timestamp',
    'Status',
    'subprocess_run',
    'PriorityQueue',
]

import enum
import subprocess  # nosec B404
from collections import UserList
from datetime import datetime
from typing import Generic, Iterable, Mapping, TypeVar

T = TypeVar('T', bound=enum.Enum)


def get_timestamp() -> str:
    timestamp = datetime.now().astimezone().isoformat()
    timestamp = timestamp.replace(':', '-')
    timestamp = timestamp.replace('+', '-')
    timestamp = timestamp.replace('.', '_')
    return timestamp


class Status(Generic[T]):

    def __init__(self, status: T) -> None:
        self._status = status

    @property
    def status(self) -> T:
        return self._status

    def transit(self, transitions: Mapping[T, T]) -> None:
        if self._status not in transitions:
            raise RuntimeError(f"{self._status} is not in {transitions}.")
        self._status = transitions[self._status]


def subprocess_run(args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,  # nosec B602
        text=True,
    ).stdout


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
