__all__ = [
    'ImageData',
]

import enum
from typing import TypeVar

from todd.utils import ArgsKwargs

from .interleaved_data import Codebook, InterleavedData

T = TypeVar('T', bound=enum.Enum)


class ImageData(InterleavedData[T]):

    def __init__(
        self,
        *args,
        image_token_type: T,
        image_wh: tuple[int, int],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._image_token_type = image_token_type
        self._image_wh = image_wh

    def __getstate__(self) -> ArgsKwargs:
        args, kwargs = super().__getstate__()
        kwargs.update(
            image_token_type=self._image_token_type,
            image_wh=self._image_wh,
        )
        return args, kwargs

    @property
    def image_codebook(self) -> Codebook[T]:
        return self._codebooks[self._image_token_type]

    @property
    def image_wh(self) -> tuple[int, int]:
        return self._image_wh
