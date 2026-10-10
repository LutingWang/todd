__all__ = [
    'DirectoryAccessLayer',
]

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, TypeVar

from .base import BaseAccessLayer

T = TypeVar('T')


class DirectoryAccessLayer(BaseAccessLayer[str, T], ABC):

    def __init__(self, *args, directory: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._directory = Path(directory)

    @abstractmethod
    def _paths(self) -> Iterator[Path]:
        pass

    def _path(self, key: str) -> Path:
        return self._directory / key

    def __iter__(self) -> Iterator[str]:
        for path in self._paths():
            yield str(path.relative_to(self._directory))

    def __len__(self) -> int:
        return len(list(self._paths()))

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def exists(self) -> bool:
        return self._directory.exists()

    def touch(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
