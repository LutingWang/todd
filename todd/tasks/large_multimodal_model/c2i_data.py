__all__ = [
    'C2IEnum',
    'C2IData',
]

import enum

import einops
import einops.layers.torch
import torch

from .interleaved_data import Codebook, Segment
from .x2i_data import X2IData


class C2IEnum(enum.Enum):
    CATEGORY = enum.auto()
    IMAGE = enum.auto()


class C2IData(X2IData[C2IEnum]):

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

        super().__init__(
            category_tokens,
            image_tokens,
            C2IEnum.CATEGORY,
            C2IEnum.IMAGE,
            category_codebook_size,
            image_codebook_size,
            *args,
            **kwargs,
        )

    @property
    def category_segment(self) -> Segment[C2IEnum]:
        return self.condition_segment

    @property
    def category_codebook(self) -> Codebook[C2IEnum]:
        return self.condition_codebook
