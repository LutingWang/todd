__all__ = [
    'FileAccessLayer',
]

import pathlib
from abc import ABC
from typing import Iterator, TypeVar

from .directory import DirectoryAccessLayer

T = TypeVar('T')


class FileAccessLayer(DirectoryAccessLayer[T], ABC):

    def __init__(
        self,
        *args,
        recursive: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._recursive = recursive

    def _paths(self) -> Iterator[pathlib.Path]:
        return filter(
            pathlib.Path.is_file,
            self._directory.rglob('*')
            if self._recursive else self._directory.iterdir(),
        )

    def __delitem__(self, key: str) -> None:
        self._path(key).unlink()
