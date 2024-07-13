__all__ = [
    'ClassConditionalImageEnum',
    'ClassConditionalImageData',
]

import enum

import einops
import einops.layers.torch
import torch

from .interleaved_data import Codebook, InterleavedData, Segment


class ClassConditionalImageEnum(enum.Enum):
    CATEGORY = enum.auto()
    IMAGE = enum.auto()


class ClassConditionalImageData(InterleavedData[ClassConditionalImageEnum]):

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
            [
                ClassConditionalImageEnum.CATEGORY,
                ClassConditionalImageEnum.IMAGE,
            ],
            {
                ClassConditionalImageEnum.CATEGORY: category_codebook_size,
                ClassConditionalImageEnum.IMAGE: image_codebook_size,
            },
            *args,
            **kwargs,
        )

        self._image_wh = (h, w)

    @property
    def condition_segment(self) -> Segment[ClassConditionalImageEnum]:
        segment = self._segments[0]
        assert segment.token_type is ClassConditionalImageEnum.CATEGORY
        return segment

    @property
    def image_segment(self) -> Segment[ClassConditionalImageEnum]:
        segment = self._segments[1]
        assert segment.token_type is ClassConditionalImageEnum.IMAGE
        return segment

    @property
    def category_codebook(self) -> Codebook[ClassConditionalImageEnum]:
        return self._codebooks[ClassConditionalImageEnum.CATEGORY]

    @property
    def image_codebook(self) -> Codebook[ClassConditionalImageEnum]:
        return self._codebooks[ClassConditionalImageEnum.IMAGE]

    @property
    def image_wh(self) -> tuple[int, int]:
        return self._image_wh
