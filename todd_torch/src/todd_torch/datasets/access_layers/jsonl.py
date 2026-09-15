__all__ = [
    'JSONLAccessLayer',
]

from typing import TypeVar

from todd import jsonl_dump, jsonl_load

from ..registries import AccessLayerRegistry
from .folder import FolderAccessLayer
from .suffix import SuffixMixin

T = TypeVar('T')


@AccessLayerRegistry.register_()
class JSONLAccessLayer(SuffixMixin[list[T]], FolderAccessLayer[list[T]]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, suffix='jsonl', **kwargs)

    def __getitem__(self, key: str) -> list[T]:
        return jsonl_load(self._file(key))

    def __setitem__(self, key: str, value: list[T]) -> None:
        jsonl_dump(value, self._file(key))
