__all__ = [
    'SupportsWrite',
]

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsWrite(Protocol):

    def write(self, s: Any) -> int:
        ...
