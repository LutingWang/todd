__all__ = [
    'PthAccessLayer',
]

import pathlib
from typing import Iterator, TypeVar

import torch

from ..registries import AccessLayerRegistry
from .folder import FolderAccessLayer

VT = TypeVar('VT')


@AccessLayerRegistry.register_()
class PthAccessLayer(FolderAccessLayer[VT]):

    def _files(self) -> Iterator[pathlib.Path]:
        return self.folder_root.glob('*.pth')

    def _file(self, key: str) -> pathlib.Path:
        return super()._file(f'{key}.pth')

    def __iter__(self) -> Iterator[str]:
        return (path.stem for path in self._files())

    def __getitem__(self, key: str) -> VT:
        return torch.load(self._file(key), map_location='cpu')

    def __setitem__(self, key: str, value: VT) -> None:
        torch.save(value, self._file(key))
