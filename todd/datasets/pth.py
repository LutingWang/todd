__all__ = [
    'PthAccessLayer',
    'PthDataset',
]

import pathlib
from typing import Iterator

import torch

from .base import (
    AccessLayerRegistry,
    BaseAccessLayer,
    BaseDataset,
    DatasetRegistry,
)


@AccessLayerRegistry.register()
class PthAccessLayer(BaseAccessLayer[str]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._path = pathlib.Path(self._data_root) / self._task_name
        if not self._readonly and not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)

    def __iter__(self) -> Iterator[str]:
        return (path.stem for path in self._path.glob('*.pth'))

    def __len__(self) -> int:
        return len(list(self._path.glob('*.pth')))

    def __getitem__(self, key: str):
        file = self._file(key, check=True)
        return torch.load(file, map_location='cpu')

    def __setitem__(self, key: str, value) -> None:
        file = self._file(key)
        torch.save(value, file)

    def __delitem__(self, key: str) -> None:
        self._file(key, check=True).unlink()

    def _file(self, key: str, check: bool = False) -> pathlib.Path:
        file = self._path / f'{key}.pth'
        if not file.exists():
            raise KeyError(key)
        return file


@DatasetRegistry.register()
class PthDataset(BaseDataset[str]):
    pass
