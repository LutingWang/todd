__all__ = [
    'X2IData',
]

import enum
from typing import TypeVar

import einops
import einops.layers.torch
import torch

from todd.utils import ArgsKwargs

from .image_data import ImageData
from .interleaved_data import Codebook, Segment

T = TypeVar('T', bound=enum.Enum)


class X2IData(ImageData[T]):

    def __init__(
        self,
        condition_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        condition_codebook_size: int,
        image_codebook_size: int,
        *args,
        condition_token_type: T,
        image_token_type: T,
        **kwargs,
    ) -> None:
        _, h, w = image_tokens.shape
        image_tokens = einops.rearrange(image_tokens, 'b h w -> b (h w)')
        super().__init__(
            [condition_tokens, image_tokens],
            [condition_token_type, image_token_type],
            {
                condition_token_type: condition_codebook_size,
                image_token_type: image_codebook_size,
            },
            *args,
            image_token_type=image_token_type,
            image_wh=(w, h),
            **kwargs,
        )
        self._condition_token_type = condition_token_type

    def __getstate__(self) -> ArgsKwargs:
        args, kwargs = super().__getstate__()
        tokens, _, codebook_sizes, *args_ = args
        args = tuple(args_)
        condition_tokens, image_tokens = tokens
        image_tokens = einops.rearrange(
            image_tokens,
            'b (h w) -> b h w',
            h=self._image_wh[1],
            w=self._image_wh[0],
        )
        args = (
            condition_tokens,
            image_tokens,
            codebook_sizes[self._condition_token_type],
            codebook_sizes[self._image_token_type],
        ) + args

        kwargs['condition_token_type'] = self._condition_token_type
        kwargs.pop('image_wh')
        return args, kwargs

    @property
    def condition_segment(self) -> Segment[T]:
        segment = self._segments[0]
        assert segment.token_type is self._condition_token_type
        return segment

    @property
    def image_segment(self) -> Segment[T]:
        segment = self._segments[1]
        assert segment.token_type is self._image_token_type
        return segment

    @property
    def condition_codebook(self) -> Codebook[T]:
        return self._codebooks[self._condition_token_type]
