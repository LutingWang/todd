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

    def __init__(self, *args, suffix: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._suffix = '.' + suffix

    def _files(self) -> Iterator[pathlib.Path]:
        files = super()._files()
        return filter(lambda file: file.suffix == self._suffix, files)

    def _file(self, key: str) -> pathlib.Path:
        return super()._file(key + self._suffix)

    def __iter__(self) -> Iterator[str]:
        return map(
            lambda file: file.removesuffix(self._suffix),
            super().__iter__(),
        )
