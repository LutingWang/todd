__all__ = [
    'Status',
]

import enum
from typing import Generic, Mapping, TypeVar

T = TypeVar('T', bound=enum.Enum)


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
