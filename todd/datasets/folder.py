__all__ = [
    'FolderAccessLayer',
]

import pathlib
from typing import Iterator, TypeVar

from ..base import Config
from .base import BaseAccessLayer

KT = TypeVar('KT')
VT = TypeVar('VT')


class FolderAccessLayer(BaseAccessLayer[str, VT]):

    def __init__(
        self,
        *args,
        folder_root: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if folder_root is None:
            folder_root = Config()
        self._build_folder_root(folder_root)

    def _build_folder_root(self, config: Config) -> None:
        self._folder_root = pathlib.Path(self._data_root) / self._task_name

    @property
    def exists(self) -> bool:
        return self._folder_root.exists()

    def touch(self) -> None:
        self._folder_root.mkdir(parents=True, exist_ok=True)

    def _files(self) -> Iterator[pathlib.Path]:
        return (path for path in self._folder_root.iterdir() if path.is_file())

    def _file(self, key: str) -> pathlib.Path:
        return self._folder_root / key

    def __iter__(self) -> Iterator[str]:
        return (path.name for path in self._files())

    def __len__(self) -> int:
        return len(list(self._files()))

    def __delitem__(self, key: str) -> None:
        self._file(key).unlink()
