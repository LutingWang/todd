__all__ = [
    'BaseAccessLayer',
]

from abc import abstractmethod
from typing import MutableMapping, TypeVar

KT = TypeVar('KT')
VT = TypeVar('VT')


class BaseAccessLayer(MutableMapping[KT, VT]):

    def __init__(
        self,
        data_root: str,
        task_name: str = '',
    ) -> None:
        self._data_root = data_root
        self._task_name = task_name

    @property
    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def touch(self) -> None:
        pass
