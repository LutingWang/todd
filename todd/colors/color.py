__all__ = [
    'Color',
]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import Self

if TYPE_CHECKING:
    from .rgba import RGBA

T = TypeVar('T', bound='Color')


class Color(ABC):

    @property
    def red(self) -> float:
        return self._to().red

    @property
    def green(self) -> float:
        return self._to().green

    @property
    def blue(self) -> float:
        return self._to().blue

    @property
    def alpha(self) -> float:
        return self._to().alpha

    @property
    def luminance(self) -> float:
        from .yiq import YIQ
        return self.to(YIQ).luminance

    @property
    def in_phase(self) -> float:
        from .yiq import YIQ
        return self.to(YIQ).in_phase

    @property
    def quadrature(self) -> float:
        from .yiq import YIQ
        return self.to(YIQ).quadrature

    @classmethod
    def _normalize(cls, value: float) -> float:
        return value / 255 if isinstance(value, int) else value

    @classmethod
    def _denormalize(cls, value: float) -> int:
        return int(value * 255)

    @classmethod
    @abstractmethod
    def _from(cls, rgba: 'RGBA') -> Self:
        pass

    @classmethod
    def from_(cls, color: 'Color') -> Self:
        if isinstance(color, cls):
            return color
        return cls._from(color._to())

    @abstractmethod
    def _to(self) -> 'RGBA':
        pass

    def to(self, cls: type[T]) -> T:
        if isinstance(self, cls):
            return self
        return cls._from(self._to())

    @abstractmethod
    def _to_tuple(self) -> tuple[float, ...]:
        pass

    def to_tuple(self, normalized: bool = False) -> tuple[float, ...]:
        tuple_ = self._to_tuple()
        if normalized:
            return tuple_
        return tuple(map(self._denormalize, tuple_))
