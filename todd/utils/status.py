__all__ = [
    'Status',
]

import enum
from typing import Generic, Mapping, TypeVar

T = TypeVar('T', bound=enum.Enum)


class Status(Generic[T]):
    """A class to represent a status that can be transited to another status.

    Examples:
        >>> class Enum(enum.Enum):
        ...     A = 1
        ...     B = 2
        ...     C = 3
        >>> status = Status(Enum.A)
        >>> status.status
        <Enum.A: 1>
        >>> status.transit({Enum.A: Enum.B})
        >>> status.status
        <Enum.B: 2>
        >>> status.transit({Enum.A: Enum.C})
        Traceback (most recent call last):
            ...
        RuntimeError: Enum.B is not in {<Enum.A: 1>: <Enum.C: 3>}.
    """

    def __init__(self, status: T) -> None:
        self._status = status

    @property
    def status(self) -> T:
        return self._status

    def transit(self, transitions: Mapping[T, T]) -> None:
        if self._status not in transitions:
            raise RuntimeError(f"{self._status} is not in {transitions}.")
        self._status = transitions[self._status]
