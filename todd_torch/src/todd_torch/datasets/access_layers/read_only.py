__all__ = [
    'ReadOnlyMixin',
]

from abc import ABC
from typing import TypeVar

from .base import BaseAccessLayer

K = TypeVar('K')
V = TypeVar('V')


class ReadOnlyMixin(BaseAccessLayer[K, V], ABC):

    def touch(self) -> None:
        raise NotImplementedError

    def __setitem__(self, key: K, value: V) -> None:
        raise NotImplementedError

    def __delitem__(self, key: K) -> None:
        raise NotImplementedError
