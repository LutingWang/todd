__all__ = [
    'DirectoryAccessLayer',
]

import pathlib
from abc import ABC, abstractmethod
from typing import Iterator, TypeVar

from .base import BaseAccessLayer

T = TypeVar('T')


class DirectoryAccessLayer(BaseAccessLayer[str, T], ABC):

    def __init__(
        self,
        *args,
        directory: pathlib.Path,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._directory = directory

    @abstractmethod
    def _paths(self) -> Iterator[pathlib.Path]:
        pass

    def _path(self, key: str) -> pathlib.Path:
        return self._directory / key

    def __iter__(self) -> Iterator[str]:
        for path in self._paths():
            yield str(path.relative_to(self._directory))

    def __len__(self) -> int:
        return len(list(self._paths()))

    @property
    def directory(self) -> pathlib.Path:
        return self._directory

    @property
    def exists(self) -> bool:
        return self._directory.exists()

    def touch(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
