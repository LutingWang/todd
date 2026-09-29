__all__ = [
    'DirectoryAccessLayer',
]

import pathlib
from abc import ABC
from typing import Generator, Iterator, TypeVar

from .base import BaseAccessLayer

V = TypeVar('V')


class DirectoryAccessLayer(BaseAccessLayer[str, V], ABC):

    def __init__(
        self,
        *args,
        directory: pathlib.Path,
        recursive: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._directory = directory
        self._recursive = recursive

    @property
    def directory(self) -> pathlib.Path:
        return self._directory

    @property
    def exists(self) -> bool:
        return self._directory.exists()

    def touch(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)

    def _files(self) -> Iterator[pathlib.Path]:
        return filter(
            pathlib.Path.is_file,
            self._directory.rglob('*')
            if self._recursive else self._directory.iterdir(),
        )

    def _file(self, key: str) -> pathlib.Path:
        return self._directory / key

    def __iter__(self) -> Generator[str, None, None]:
        for path in self._files():
            yield (
                str(path.relative_to(self._directory))
                if self._recursive else path.name
            )

    def __len__(self) -> int:
        return len(list(self._files()))

    def __delitem__(self, key: str) -> None:
        self._file(key).unlink()
