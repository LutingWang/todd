__all__ = [
    'JSONAccessLayer',
]

from typing import TypeVar

from todd import json_dump, json_load

from ..registries import AccessLayerRegistry
from .folder import FolderAccessLayer
from .suffix import SuffixMixin

T = TypeVar('T')


@AccessLayerRegistry.register_()
class JSONAccessLayer(SuffixMixin[T], FolderAccessLayer[T]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, suffix='json', **kwargs)

    def __getitem__(self, key: str) -> T:
        return json_load(self._file(key))

    def __setitem__(self, key: str, value: T) -> None:
        json_dump(value, self._file(key))
