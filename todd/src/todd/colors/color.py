__all__ = [
    'Color',
]

from abc import ABC, abstractmethod
from types import NotImplementedType
from typing import TYPE_CHECKING, TypeVar, cast
from typing_extensions import Self

if TYPE_CHECKING:
    from .rgba import RGBA

T = TypeVar('T', bound='Color')


class Color(ABC):

    @property
    def red(self) -> float:
        from .rgba import RGB
        return self.to(RGB).red

    @property
    def green(self) -> float:
        from .rgba import RGB
        return self.to(RGB).green

    @property
    def blue(self) -> float:
        from .rgba import RGB
        return self.to(RGB).blue

    @property
    def alpha(self) -> float:
        from .rgba import RGBA
        return self.to(RGBA).alpha

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
    def _from(cls, color: 'Color') -> Self | NotImplementedType:
        return NotImplemented

    @classmethod
    @abstractmethod
    def _from_rgba(cls, rgba: 'RGBA') -> Self:
        pass

    @classmethod
    def from_(cls, color: 'Color') -> Self:
        if color.__class__ is cls:
            return cast(Self, color)
        color_ = cls._from(color)
        if color_ is not NotImplemented:
            return color_
        return color.to(cls)

    def _to(self, cls: type[T]) -> T | NotImplementedType:
        return NotImplemented

    @abstractmethod
    def _to_rgba(self) -> 'RGBA':
        pass

    def to(self, cls: type[T]) -> T:
        if self.__class__ is cls:
            return cast(T, self)
        color = self._to(cls)
        if color is not NotImplemented:
            return color
        color = cls._from(self)
        if color is not NotImplemented:
            return color
        return cls._from_rgba(self._to_rgba())

    @abstractmethod
    def to_tuple(self) -> tuple[float, ...]:
        pass
