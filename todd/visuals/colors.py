__all__ = [
    'Color',
    'RGB',
    'RGBA',
    'BGR',
    'YIQ',
    'HTML4',
    'PALETTE',
]

import enum
from abc import ABC, abstractmethod
from colorsys import rgb_to_yiq, yiq_to_rgb
from typing import TypeVar
from typing_extensions import Self

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
        return self.to(YIQ).luminance

    @property
    def in_phase(self) -> float:
        return self.to(YIQ).in_phase

    @property
    def quadrature(self) -> float:
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


class RGB(Color):

    def __init__(self, red: float, green: float, blue: float) -> None:
        self._red = self._normalize(red)
        self._green = self._normalize(green)
        self._blue = self._normalize(blue)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._red}, {self._green}, {self._blue})"
        )

    @property
    def red(self) -> float:
        return self._red

    @property
    def green(self) -> float:
        return self._green

    @property
    def blue(self) -> float:
        return self._blue

    @classmethod
    def _from(cls, rgba: 'RGBA') -> Self:
        r, g, b, _ = rgba.to_tuple()
        return cls(r, g, b)

    @classmethod
    def from_(cls, color: Color | str) -> Self:
        if isinstance(color, str):
            assert len(color) == 7 and color[0] == '#'
            return cls(
                int(color[1:3], 16), int(color[3:5], 16), int(color[5:], 16)
            )
        return super().from_(color)

    def _to(self) -> 'RGBA':
        return RGBA(self._red, self._green, self._blue, alpha=1.)

    def _to_tuple(self) -> tuple[float, ...]:
        return self._red, self._green, self._blue


class RGBA(RGB):

    def __init__(self, *args, alpha: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert 0. <= alpha <= 1.
        self._alpha = alpha

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._red}, {self._green}, {self._blue}, "
            f"alpha={self._alpha})"
        )

    @property
    def alpha(self) -> float:
        return self._alpha

    @classmethod
    def _from(cls, rgba: 'RGBA') -> Self:
        *rgb, a = rgba.to_tuple()
        return cls(*rgb, alpha=a)

    def _to(self) -> 'RGBA':
        return self

    def to_tuple(self, *args, **kwargs) -> tuple[float, ...]:
        return super().to_tuple(*args, **kwargs) + (self._alpha, )


class BGR(RGB):

    def __init__(self, blue: float, green: float, red: float) -> None:
        super().__init__(red, green, blue)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._blue}, {self._green}, {self._red})"
        )


class YIQ(Color):

    def __init__(
        self,
        luminance: float,
        in_phase: float,
        quadrature: float,
    ) -> None:
        self._luminance = self._normalize(luminance)
        self._in_phase = self._normalize(in_phase)
        self._quadrature = self._normalize(quadrature)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._luminance}, {self._in_phase}, "
            f"{self._quadrature})"
        )

    @property
    def luminance(self) -> float:
        return self._luminance

    @property
    def in_phase(self) -> float:
        return self._in_phase

    @property
    def quadrature(self) -> float:
        return self._quadrature

    @classmethod
    def _from(cls, rgba: RGBA) -> Self:
        r, g, b, *_ = rgba.to_tuple()
        y, i, q = rgb_to_yiq(r, g, b)
        return cls(y, i, q)

    def _to(self) -> RGBA:
        r, g, b = yiq_to_rgb(*self._to_tuple())
        return RGBA(r, g, b, alpha=1.)

    def _to_tuple(self) -> tuple[float, ...]:
        return self._luminance, self._in_phase, self._quadrature


# https://www.w3.org/TR/html401/types.html#h-6.5
class HTML4(enum.StrEnum):
    BLACK = '#000000'
    SILVER = '#C0C0C0'
    GRAY = '#808080'
    WHITE = '#FFFFFF'
    MAROON = '#800000'
    RED = '#FF0000'
    PURPLE = '#800080'
    FUCHSIA = '#FF00FF'
    GREEN = '#008000'
    LIME = '#00FF00'
    OLIVE = '#808000'
    YELLOW = '#FFFF00'
    NAVY = '#000080'
    BLUE = '#0000FF'
    TEAL = '#008080'
    AQUA = '#00FFFF'


PALETTE = [RGB.from_(color.value) for color in HTML4]
