__all__ = [
    'SubdirectoryAccessLayer',
]

import pathlib
from abc import ABC
from shutil import rmtree
from typing import Iterator, TypeVar

from .directory import DirectoryAccessLayer

T = TypeVar('T')


class SubdirectoryAccessLayer(DirectoryAccessLayer[T], ABC):

    def _paths(self) -> Iterator[pathlib.Path]:
        return filter(pathlib.Path.is_dir, self._directory.iterdir())

    def __delitem__(self, key: str) -> None:
        rmtree(self._path(key))
