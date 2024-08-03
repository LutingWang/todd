__all__ = [
    'SuffixMixin',
]

import pathlib
from typing import Iterator, TypeVar

from ..registries import AccessLayerRegistry
from .folder import FolderAccessLayer

VT = TypeVar('VT')


@AccessLayerRegistry.register_()
class SuffixMixin(FolderAccessLayer[VT]):

    def __init__(self, *args, suffix: str | None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if suffix is not None:
            self._suffix = '.' + suffix

    @property
    def with_suffix(self) -> bool:
        return hasattr(self, '_suffix')

    def _files(self) -> Iterator[pathlib.Path]:
        files = super()._files()
        if self.with_suffix:
            files = filter(lambda file: file.suffix == self._suffix, files)
        return files

    def _file(self, key: str) -> pathlib.Path:
        if self.with_suffix:
            return super()._file(key + self._suffix)
        return super()._file(key)

    def __iter__(self) -> Iterator[str]:
        iter_ = super().__iter__()
        if self.with_suffix:
            iter_ = map(lambda file: file.removesuffix(self._suffix), iter_)
        return iter_
