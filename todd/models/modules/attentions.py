__all__ = [
    'BaseAttention',
]

from abc import ABC

from torch import nn


class BaseAttention(nn.Module, ABC):

    def __init__(self, *args, width: int, num_heads: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._width = width
        self._num_heads = num_heads

    @property
    def head_dim(self) -> int:
        return self._width // self._num_heads

    @property
    def hidden_dim(self) -> int:
        return self.head_dim * self._num_heads
