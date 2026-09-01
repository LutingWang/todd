__all__ = [
    'Segment',
    'Codebook',
    'InterleavedData',
]

import enum
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, Iterable, Mapping, TypeVar

import torch

from todd.utils import ArgsKwargs, SerializeMixin

T = TypeVar('T', bound=enum.Enum)


@dataclass(frozen=True)
class Segment(Generic[T]):
    tokens: torch.Tensor
    token_type: T
    start: int

    @property
    def length(self) -> int:
        return self.tokens.shape[1]

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def middle(self) -> int:
        return self.start + self.length // 2

    @property
    def slice_(self) -> slice:
        return slice(self.start, self.end)


@dataclass(frozen=True)
class Codebook(Generic[T]):
    token_type: T
    size: int
    bias: int

    @property
    def start(self) -> int:
        return self.bias

    @property
    def end(self) -> int:
        return self.bias + self.size

    def bias_(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.bias

    def debias(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens - self.bias


class InterleavedData(SerializeMixin, Generic[T]):

    def __init__(
        self,
        tokens: Iterable[torch.Tensor],
        token_types: Iterable[T],
        codebook_sizes: Mapping[T, int],
        share_codebook: bool = False,
    ) -> None:
        segments: list[Segment[T]] = []
        for token, token_type in zip(tokens, token_types):
            segment_start = segments[-1].end if segments else 0
            segment = Segment(token, token_type, segment_start)
            segments.append(segment)
        self._segments = tuple(segments)

        codebooks: list[Codebook[T]] = []
        for codebook_type, codebook_size in codebook_sizes.items():
            codebook_start = (
                0
                if share_codebook or len(codebooks) == 0 else codebooks[-1].end
            )
            codebook = Codebook(codebook_type, codebook_size, codebook_start)
            codebooks.append(codebook)
        self._codebooks = {
            codebook.token_type: codebook
            for codebook in codebooks
        }

        self._share_codebook = share_codebook

    def __getstate__(self) -> ArgsKwargs:
        tokens = [segment.tokens for segment in self._segments]
        token_types = [segment.token_type for segment in self._segments]
        codebook_sizes = {
            token_type: codebook.size
            for token_type, codebook in self._codebooks.items()
        }
        args = (tokens, token_types, codebook_sizes)
        share_codebook = self._share_codebook
        kwargs = dict(share_codebook=share_codebook)
        return args, kwargs

    def __len__(self) -> int:
        return self.tokens.shape[1]

    @cached_property
    def tokens(self) -> torch.Tensor:
        tokens = [
            self._codebooks[segment.token_type].bias_(segment.tokens)
            for segment in self._segments
        ]
        return torch.cat(tokens, -1)

    @property
    def codebooks(self) -> dict[T, Codebook[T]]:
        return self._codebooks

    @property
    def segments(self) -> tuple[Segment[T], ...]:
        return self._segments
