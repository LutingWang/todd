__all__ = [
    'Status',
]

import enum
from typing import Generic, Mapping, TypeVar

EnumType = TypeVar('EnumType', bound=enum.Enum)  # pylint: disable=invalid-name


class Status(Generic[EnumType]):

    def __init__(self, status: EnumType) -> None:
        self._status = status

    @property
    def status(self) -> EnumType:
        return self._status

    def transit(self, transitions: Mapping[EnumType, EnumType]) -> None:
        if self._status not in transitions:
            raise RuntimeError(f"{self._status} is not in {transitions}.")
        self._status = transitions[self._status]
