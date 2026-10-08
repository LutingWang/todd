__all__ = [
    'NpyAccessLayer',
]

from typing import TypeVar

import numpy as np
import numpy.typing as npt

from ..registries import AccessLayerRegistry
from .file import FileAccessLayer
from .suffix import SuffixMixin

T = TypeVar('T', bound=np.number)


@AccessLayerRegistry.register_()
class NpyAccessLayer(
    SuffixMixin[npt.NDArray[T]],
    FileAccessLayer[npt.NDArray[T]],
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, suffix='.npy', **kwargs)

    def __getitem__(self, key: str) -> npt.NDArray[T]:
        return np.load(self._path(key))

    def __setitem__(self, key: str, value: npt.NDArray[T]) -> None:
        np.save(self._path(key), value)
