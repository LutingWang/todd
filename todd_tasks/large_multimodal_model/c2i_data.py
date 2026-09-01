__all__ = [
    'C2IEnum',
    'C2IData',
]

import enum

import einops
import einops.layers.torch
import torch

from todd.utils import ArgsKwargs

from .interleaved_data import Codebook, Segment
from .x2i_data import X2IData


class C2IEnum(enum.Enum):
    CATEGORY = enum.auto()
    IMAGE = enum.auto()


class C2IData(X2IData[C2IEnum]):

    def __init__(self, category_tokens: torch.Tensor, *args, **kwargs) -> None:
        if category_tokens.dim() == 1:
            category_tokens = einops.rearrange(category_tokens, 'b -> b 1')
        assert category_tokens.dim() == 2

        super().__init__(
            category_tokens,
            *args,
            condition_token_type=C2IEnum.CATEGORY,
            image_token_type=C2IEnum.IMAGE,
            **kwargs,
        )

    def __getstate__(self) -> ArgsKwargs:
        args, kwargs = super().__getstate__()
        kwargs.pop('condition_token_type')
        kwargs.pop('image_token_type')
        return args, kwargs

    @property
    def category_segment(self) -> Segment[C2IEnum]:
        return self.condition_segment

    @property
    def category_codebook(self) -> Codebook[C2IEnum]:
        return self.condition_codebook
