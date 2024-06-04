__all__ = [
    'PthAccessLayer',
]

from typing import TypeVar

import torch

from ..registries import AccessLayerRegistry
from .folder import FolderAccessLayer
from .suffix import SuffixMixin

VT = TypeVar('VT')


@AccessLayerRegistry.register_()
class PthAccessLayer(SuffixMixin[VT], FolderAccessLayer[VT]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, suffix='pth', **kwargs)

    def __getitem__(self, key: str) -> VT:
        return torch.load(self._file(key), map_location='cpu')

    def __setitem__(self, key: str, value: VT) -> None:
        torch.save(value, self._file(key))
