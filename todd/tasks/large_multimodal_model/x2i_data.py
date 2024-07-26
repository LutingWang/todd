__all__ = [
    'X2IData',
]

import enum
from typing import TypeVar
from typing_extensions import Self

import einops
import einops.layers.torch
import torch

from todd.utils import ArgsKwargs

from .interleaved_data import Codebook, InterleavedData, Segment

T = TypeVar('T', bound=enum.Enum)


class X2IData(InterleavedData[T]):

    def __init__(
        self,
        condition_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        condition_token_type: T,
        image_token_type: T,
        condition_codebook_size: int,
        image_codebook_size: int,
        *args,
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
            **kwargs,
        )
        self._condition_token_type = condition_token_type
        self._image_token_type = image_token_type
        self._image_wh = (h, w)

    def __getstate__(self) -> ArgsKwargs:
        args, kwargs = super().__getstate__()
        tokens, token_types, codebook_sizes, *args = args  # type: ignore[assignment] # noqa: E501 pylint: disable=line-too-long
        condition_tokens, image_tokens = tokens
        condition_token_type, image_token_type = token_types
        condition_codebook_size = codebook_sizes[condition_token_type]
        image_codebook_size = codebook_sizes[image_token_type]

        image_tokens = einops.rearrange(
            self.image_segment.tokens,
            'b (h w) -> b h w',
            h=self._image_wh[1],
            w=self._image_wh[0],
        )
        args = (
            condition_tokens,
            image_tokens,
            condition_token_type,
            image_token_type,
            condition_codebook_size,
            image_codebook_size,
        ) + args
        return args, kwargs

    def dropout(self, condition_tokens: torch.Tensor) -> Self:
        args, kwargs = self.__getstate__()
        return self.__class__(condition_tokens, *args[1:], **kwargs)

    def cat(self, other: Self) -> Self:
        args, kwargs = self.__getstate__()
        condition_tokens, image_tokens, *args = args  # type: ignore[assignment] # noqa: E501 pylint: disable=line-too-long
        other_args, other_kwargs = other.__getstate__()
        other_condition_tokens, other_image_tokens, *other_args = other_args  # type: ignore[assignment] # noqa: E501 pylint: disable=line-too-long
        condition_tokens = torch.cat([
            condition_tokens,
            other_condition_tokens,
        ])
        image_tokens = torch.cat([image_tokens, other_image_tokens])
        assert args == other_args
        assert kwargs == other_kwargs
        return self.__class__(condition_tokens, image_tokens, *args, **kwargs)

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

    @property
    def image_codebook(self) -> Codebook[T]:
        return self._codebooks[self._image_token_type]

    @property
    def image_wh(self) -> tuple[int, int]:
        return self._image_wh
