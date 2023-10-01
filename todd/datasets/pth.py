__all__ = [
    'PthAccessLayer',
    'PthDataset',
]

import pathlib
from typing import Iterator, TypeVar

import torch

from .base import AccessLayerRegistry, BaseDataset
from .folder import FolderAccessLayer

T = TypeVar('T')
VT = TypeVar('VT')


@AccessLayerRegistry.register()
class PthAccessLayer(FolderAccessLayer[VT]):

    def _files(self) -> Iterator[pathlib.Path]:
        return self._folder_root.glob('*.pth')

    def _file(self, key: str) -> pathlib.Path:
        return super()._file(f'{key}.pth')

    def __iter__(self) -> Iterator[str]:
        return (path.stem for path in self._files())

    def __getitem__(self, key: str) -> VT:
        return torch.load(self._file(key), map_location='cpu')

    def __setitem__(self, key: str, value: VT) -> None:
        torch.save(value, self._file(key))


class PthDataset(BaseDataset[T, str, VT]):
    ACCESS_LAYER = PthAccessLayer.__name__
