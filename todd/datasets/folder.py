__all__ = [
    'FolderAccessLayer',
]

import itertools
import pathlib
from abc import ABC
from typing import Iterator, TypeVar

from ..base import Config
from .base import BaseAccessLayer

VT = TypeVar('VT')


class FolderAccessLayer(BaseAccessLayer[str, VT], ABC):

    def __init__(
        self,
        *args,
        folder_root: Config | None = None,
        filter_directories: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if folder_root is None:
            folder_root = Config()
        self._build_folder_root(folder_root)

        self._filter_directories = filter_directories

    def _build_folder_root(self, config: Config) -> None:
        self._folder_root = pathlib.Path(self._data_root) / self._task_name

    @property
    def exists(self) -> bool:
        return self._folder_root.exists()

    def touch(self) -> None:
        self._folder_root.mkdir(parents=True, exist_ok=True)

    def _files(self) -> Iterator[pathlib.Path]:
        files: Iterator[pathlib.Path] = self._folder_root.iterdir()
        if self._filter_directories:
            files = itertools.filterfalse(
                lambda path: path.is_dir(),
                files,
            )
        return files

    def _file(self, key: str) -> pathlib.Path:
        return self._folder_root / key

    def __iter__(self) -> Iterator[str]:
        return map(lambda path: path.name, self._files())

    def __len__(self) -> int:
        return len(list(self._files()))

    def __delitem__(self, key: str) -> None:
        self._file(key).unlink()
