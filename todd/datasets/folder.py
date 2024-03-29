__all__ = [
    'FolderAccessLayer',
]

import enum
import pathlib
from abc import ABC
from typing import Iterator, TypeVar

from ..base import Config
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
        folder_root: Config | None = None,
        subfolder_action: str | Action = Action.NONE,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if folder_root is None:
            folder_root = Config()
        self._build_folder_root(folder_root)

        if isinstance(subfolder_action, str):
            subfolder_action = Action(subfolder_action.lower())
        self._subfolder_action = subfolder_action

    def _build_folder_root(self, config: Config) -> None:
        self._folder_root = pathlib.Path(self._data_root) / self._task_name

    @property
    def exists(self) -> bool:
        return self._folder_root.exists()

    def touch(self) -> None:
        self._folder_root.mkdir(parents=True, exist_ok=True)

    def _files(self) -> Iterator[pathlib.Path]:
        files: Iterator[pathlib.Path]
        if self._subfolder_action is Action.WALK:
            files = self._folder_root.rglob('*')
        else:
            files = self._folder_root.iterdir()
        if self._subfolder_action in [Action.WALK, Action.FILTER]:
            files = filter(
                lambda path: path.is_file(),
                files,
            )
        return files

    def _file(self, key: str) -> pathlib.Path:
        return self._folder_root / key

    def _name(self, path: pathlib.Path) -> str:
        return path.name

    def _relative_to(self, path: pathlib.Path) -> str:
        return str(path.relative_to(self._folder_root))

    def __iter__(self) -> Iterator[str]:
        if self._subfolder_action in [Action.NONE, Action.FILTER]:
            func = self._name
        elif self._subfolder_action is Action.WALK:
            func = self._relative_to
        else:
            raise NotImplementedError
        return map(func, self._files())

    def __len__(self) -> int:
        return len(list(self._files()))

    def __delitem__(self, key: str) -> None:
        self._file(key).unlink()
