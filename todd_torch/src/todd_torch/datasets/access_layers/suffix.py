__all__ = [
    'SuffixMixin',
]

import pathlib
from typing import Iterator, TypeVar

from ..registries import AccessLayerRegistry
from .file import FileAccessLayer

V = TypeVar('V')


@AccessLayerRegistry.register_()
class SuffixMixin(FileAccessLayer[V]):

    def __init__(self, *args, suffix: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert not suffix or suffix.startswith('.')
        self._suffix = suffix

    def _paths(self) -> Iterator[pathlib.Path]:
        paths = super()._paths()
        if self._suffix:
            paths = filter(lambda path: path.suffix == self._suffix, paths)
        return paths

    def _path(self, key: str) -> pathlib.Path:
        if self._suffix:
            return super()._path(key + self._suffix)
        return super()._path(key)

    def __iter__(self) -> Iterator[str]:
        iter_ = super().__iter__()
        if self._suffix:
            iter_ = map(lambda key: key.removesuffix(self._suffix), iter_)
        return iter_
