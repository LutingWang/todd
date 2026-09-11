__all__ = [
    'RGB',
    'RGBA',
]

from typing import Literal, Sequence
from typing_extensions import Self

from .color import Color


def normalize(value: float) -> float:
    return value / 255


def denormalize(value: float) -> int:
    return int(value * 255)


class RGB(Color):

    def __init__(
        self,
        *args,
        red: float,
        green: float,
        blue: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert all(0. <= channel <= 1. for channel in (red, green, blue))
        self._red = red
        self._green = green
        self._blue = blue

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._red}, {self._green}, "
            f"{self._blue})"
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
    def from_hex(cls, color: str) -> Self:
        assert len(color) == 7 and color[0] == '#'
        return cls(
            red=normalize(int(color[1:3], 16)),
            green=normalize(int(color[3:5], 16)),
            blue=normalize(int(color[5:], 16)),
        )

    @classmethod
    def from_tuple(
        cls,
        tuple_: Sequence[float],
        normalized: bool = True,
        order: Literal['rgb', 'bgr'] = 'rgb',
    ) -> Self:
        if not normalized:
            tuple_ = tuple(map(normalize, tuple_))
        if order == 'rgb':
            red, green, blue = tuple_
        elif order == 'bgr':
            blue, green, red = tuple_
        else:
            raise ValueError(f'Invalid order: {order}')
        return cls(red=red, green=green, blue=blue)

    @classmethod
    def _from_rgba(cls, rgba: 'RGBA') -> Self:
        return cls(red=rgba.red, green=rgba.green, blue=rgba.blue)

    def _to_rgba(self) -> 'RGBA':
        return RGBA(
            red=self.red,
            green=self.green,
            blue=self.blue,
            alpha=1,
        )

    def _to_tuple(
        self,
        order: Literal['rgb', 'bgr'],
    ) -> tuple[float, ...]:
        if order == 'rgb':
            return self._red, self._green, self._blue
        if order == 'bgr':
            return self._blue, self._green, self._red
        raise ValueError(f'Invalid order: {order}')

    def to_tuple(
        self,
        normalized: bool = True,
        order: Literal['rgb', 'bgr'] = 'rgb',
    ) -> tuple[float, ...]:
        tuple_ = self._to_tuple(order)
        if normalized:
            return tuple_
        return tuple(map(denormalize, tuple_))

    def to_css(self) -> str:
        red, green, blue = self.to_tuple(normalized=False)
        return f'rgb({red},{green},{blue})'


class RGBA(RGB):

    def __init__(self, *args, alpha: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert 0. <= alpha <= 1.
        self._alpha = alpha

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._red}, {self._green}, "
            f"{self._blue}, alpha={self._alpha})"
        )

    @property
    def alpha(self) -> float:
        return self._alpha

    @classmethod
    def _from_rgba(cls, rgba: 'RGBA') -> Self:
        return cls(
            red=rgba.red,
            green=rgba.green,
            blue=rgba.blue,
            alpha=rgba.alpha,
        )

    def _to_rgba(self) -> 'RGBA':
        return self

    def _to_tuple(self, *args, **kwargs) -> tuple[float, ...]:
        return super()._to_tuple(*args, **kwargs) + (self._alpha, )

    def to_css(self) -> str:
        red, green, blue, *_ = self.to_tuple(normalized=False)
        return f'rgba({red},{green},{blue},{self.alpha:g})'
