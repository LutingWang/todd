__all__ = [
    'C2IEnum',
    'C2IData',
]

import enum

import einops
import einops.layers.torch
import torch

from .interleaved_data import Codebook, InterleavedData, Segment


class C2IEnum(enum.Enum):
    CATEGORY = enum.auto()
    IMAGE = enum.auto()


class C2IData(InterleavedData[C2IEnum]):

    def __init__(
        self,
        category_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        category_codebook_size: int,
        image_codebook_size: int,
        *args,
        **kwargs,
    ) -> None:
        if category_tokens.ndim == 1:
            category_tokens = einops.rearrange(category_tokens, 'b -> b 1')
        assert category_tokens.ndim == 2

        _, h, w = image_tokens.shape
        image_tokens = einops.rearrange(image_tokens, 'b h w -> b (h w)')

        super().__init__(
            [category_tokens, image_tokens],
            [C2IEnum.CATEGORY, C2IEnum.IMAGE],
            {
                C2IEnum.CATEGORY: category_codebook_size,
                C2IEnum.IMAGE: image_codebook_size,
            },
            *args,
            **kwargs,
        )

        self._image_wh = (h, w)

    @property
    def condition_segment(self) -> Segment[C2IEnum]:
        segment = self._segments[0]
        assert segment.token_type is C2IEnum.CATEGORY
        return segment

    @property
    def image_segment(self) -> Segment[C2IEnum]:
        segment = self._segments[1]
        assert segment.token_type is C2IEnum.IMAGE
        return segment

    @property
    def category_codebook(self) -> Codebook[C2IEnum]:
        return self._codebooks[C2IEnum.CATEGORY]

    @property
    def image_codebook(self) -> Codebook[C2IEnum]:
        return self._codebooks[C2IEnum.IMAGE]

    @property
    def image_wh(self) -> tuple[int, int]:
        return self._image_wh
