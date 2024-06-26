__all__ = [
    'RGB',
    'RGBA',
    'BGR',
]

from typing_extensions import Self

from .color import Color


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
                int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:], 16),
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
