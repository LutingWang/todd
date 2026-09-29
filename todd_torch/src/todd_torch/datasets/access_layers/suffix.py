__all__ = [
    'SuffixMixin',
]

import pathlib
from typing import Iterator, TypeVar

from ..registries import AccessLayerRegistry
from .directory import DirectoryAccessLayer

V = TypeVar('V')


@AccessLayerRegistry.register_()
class SuffixMixin(DirectoryAccessLayer[V]):

    def __init__(self, *args, suffix: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert not suffix or suffix.startswith('.')
        self._suffix = suffix

    def _files(self) -> Iterator[pathlib.Path]:
        files = super()._files()
        if self._suffix:
            files = filter(lambda file: file.suffix == self._suffix, files)
        return files

    def _file(self, key: str) -> pathlib.Path:
        if self._suffix:
            return super()._file(key + self._suffix)
        return super()._file(key)

    def __iter__(self) -> Iterator[str]:
        iter_ = super().__iter__()
        if self._suffix:
            iter_ = map(lambda key: key.removesuffix(self._suffix), iter_)
        return iter_
