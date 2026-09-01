# pylint: disable=self-cls-assignment

__all__ = [
    'TensorWrapper',
    'FlattenMixin',
    'NormalizeMixin',
]

import math
from abc import ABC, abstractmethod
from typing import Any, Generator, Generic, TypeVar
from typing_extensions import Self

import torch

from todd.utils import ArgsKwargs, SerializeMixin

T = TypeVar('T')

FloatTuple = tuple[float, ...]


class TensorWrapper(SerializeMixin, ABC, Generic[T]):
    OBJECT_DIMENSIONS: int

    @classmethod
    @abstractmethod
    def to_object(cls, tensor: torch.Tensor) -> T:
        """Convert a tensor to an object.

        Args:
            tensor: the tensor with `OBJECT_DIMENSIONS` dimensions.

        Returns:
            The object.
        """

    def __init__(self, tensor: torch.Tensor, *args, **kwargs) -> None:
        assert tensor.dim() > self.OBJECT_DIMENSIONS
        super().__init__(*args, **kwargs)
        self._tensor = tensor

    def __getstate__(self) -> ArgsKwargs:
        args, kwargs = super().__getstate__()
        args = (self._tensor, ) + args
        return args, kwargs

    def __iter__(self) -> Generator[T, None, None]:
        yield from map(self.to_object, self._flatten())

    def __getitem__(self, key: Any) -> Self | T:
        tensor = self._tensor[key]
        if tensor.dim() == self.OBJECT_DIMENSIONS:
            return self.to_object(tensor)
        return self.copy(tensor)

    def __add__(self, other: Any) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        (_, *self_args), self_kwargs = self.__getstate__()
        (_, *other_args), other_kwargs = other.__getstate__()
        assert (self_args, self_kwargs) == (other_args, other_kwargs)
        bboxes = torch.cat([self._tensor, other._tensor])
        return self.copy(bboxes)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._tensor.shape[:-self.OBJECT_DIMENSIONS]

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def copy(self, tensor: torch.Tensor | None = None) -> Self:
        args, kwargs = self.__getstate__()
        if tensor is not None:
            args = (tensor, ) + args[1:]
        return self.__class__(*args, **kwargs)

    def _flatten(self) -> torch.Tensor:
        return self._tensor.flatten(end_dim=-self.OBJECT_DIMENSIONS - 1)

    @abstractmethod
    def flatten(self) -> 'FlattenMixin[T]':
        pass


class FlattenMixin(TensorWrapper[T], ABC):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self._tensor.dim() == self.OBJECT_DIMENSIONS + 1

    def __len__(self) -> int:
        return self._tensor.shape[0]


class NormalizeMixin(TensorWrapper[T], ABC):

    def __init__(
        self,
        *args,
        normalized: bool = False,
        divisor: FloatTuple | None = None,
        **kwargs,
    ) -> None:
        """Initialize.

        Args:
            normalized: whether the tensors are normalized.
            divisor: the bound for the tensors.
        """
        super().__init__(*args, **kwargs)
        self._normalized = normalized
        if divisor is not None:
            self.set_divisor(divisor)

    def __getstate__(self) -> ArgsKwargs:
        args, kwargs = super().__getstate__()
        kwargs.update(
            normalized=self.normalized,
            divisor=self.divisor if self.has_divisor else None,
        )
        return args, kwargs

    @abstractmethod
    def _scale(self, ratio: FloatTuple, /) -> torch.Tensor:
        pass

    def scale(self, ratio: FloatTuple) -> Self:
        return self.copy(self._scale(ratio))

    def __mul__(self, other: FloatTuple) -> Self:
        return self.scale(other)

    def __truediv__(self, other: FloatTuple) -> Self:
        other = tuple(1 / x for x in other)
        return self.scale(other)

    @property
    def normalized(self) -> bool:
        return self._normalized

    @property
    def has_divisor(self) -> bool:
        return hasattr(self, '_divisor')

    @property
    def divisor(self) -> FloatTuple:
        return self._divisor  # type: ignore[has-type]

    def set_divisor(self, divisor: FloatTuple, override: bool = False) -> None:
        if self.has_divisor and not override:
            assert self._divisor == divisor  # type: ignore[has-type]
            return
        self._divisor = divisor

    def normalize(self) -> Self:
        if self._normalized:
            return self
        self = self / self._divisor
        self._normalized = True
        return self

    def denormalize(self) -> Self:
        if not self._normalized:
            return self
        self = self * self._divisor
        self._normalized = False
        return self
