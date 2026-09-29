__all__ = [
    'PthAccessLayer',
]

from typing import TypeVar

import torch

from ..registries import AccessLayerRegistry
from .directory import DirectoryAccessLayer
from .suffix import SuffixMixin

V = TypeVar('V')


@AccessLayerRegistry.register_()
class PthAccessLayer(SuffixMixin[V], DirectoryAccessLayer[V]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, suffix='.pth', **kwargs)

    def __getitem__(self, key: str) -> V:
        return torch.load(self._file(key), map_location='cpu')

    def __setitem__(self, key: str, value: V) -> None:
        torch.save(value, self._file(key))
