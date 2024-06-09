__all__ = [
    'FolderAccessLayer',
]

import enum
import pathlib
from abc import ABC
from typing import Iterator, TypeVar

from .base import BaseAccessLayer

VT = TypeVar('VT')


class Action(enum.Enum):
    NONE = 'none'
    WALK = 'walk'
    FILTER = 'filter'


class FolderAccessLayer(BaseAccessLayer[str, VT], ABC):

    def __init__(
        self,
        *args,
        subfolder_action: str = 'none',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._subfolder_action = Action(subfolder_action.lower())

    @property
    def folder_root(self) -> pathlib.Path:
        return pathlib.Path(self._data_root) / self._task_name

    @property
    def exists(self) -> bool:
        return self.folder_root.exists()

    def touch(self) -> None:
        self.folder_root.mkdir(parents=True, exist_ok=True)

    def _files(self) -> Iterator[pathlib.Path]:
        files: Iterator[pathlib.Path]
        if self._subfolder_action is Action.WALK:
            files = self.folder_root.rglob('*')
        else:
            files = self.folder_root.iterdir()
        if self._subfolder_action in [Action.WALK, Action.FILTER]:
            files = filter(
                lambda path: path.is_file(),
                files,
            )
        return files

    def _file(self, key: str) -> pathlib.Path:
        return self.folder_root / key

    def __iter__(self) -> Iterator[str]:
        if self._subfolder_action in [Action.NONE, Action.FILTER]:

            def func(path: pathlib.Path) -> str:
                return path.name

        elif self._subfolder_action is Action.WALK:

            def func(path: pathlib.Path) -> str:
                return str(path.relative_to(self.folder_root))

        else:
            raise NotImplementedError
        return map(func, self._files())

    def __len__(self) -> int:
        return len(list(self._files()))

    def __delitem__(self, key: str) -> None:
        self._file(key).unlink()
