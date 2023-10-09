__all__ = [
    'get_timestamp',
    'Status',
]

import enum
from datetime import datetime
from typing import Generic, Mapping, TypeVar

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
