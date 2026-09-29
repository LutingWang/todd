__all__ = [
    'BaseAccessLayer',
]

from abc import abstractmethod
from typing import MutableMapping, TypeVar

K = TypeVar('K')
V = TypeVar('V')


class BaseAccessLayer(MutableMapping[K, V]):

    @property
    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def touch(self) -> None:
        pass
