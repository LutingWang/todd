__all__ = [
    'RGB',
    'RGBA',
]

from typing import Literal, Sequence
from typing_extensions import Self

from .color import Color


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
        return cls(red=rgba.red, green=rgba.green, blue=rgba.blue)

    @classmethod
    def from_tuple(
        cls,
        tuple_: Sequence[float],
        normalized: bool = True,
        order: Literal['rgb', 'bgr'] = 'rgb',
    ) -> Self:
        if order == 'rgb':
            red, green, blue = tuple_
        elif order == 'bgr':
            blue, green, red = tuple_
        else:
            raise ValueError(f'Invalid order: {order}')
        if not normalized:
            red /= 255
            green /= 255
            blue /= 255
        return cls(red=red, green=green, blue=blue)

    @classmethod
    def from_(cls, color: Color | str) -> Self:
        if isinstance(color, str):
            assert len(color) == 7 and color[0] == '#'
            return cls(
                red=int(color[1:3], 16) / 255,
                green=int(color[3:5], 16) / 255,
                blue=int(color[5:], 16) / 255,
            )
        return super().from_(color)

    def _to(self) -> 'RGBA':
        return RGBA(
            red=self._red,
            green=self._green,
            blue=self._blue,
            alpha=1.,
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
        return tuple(int(channel * 255) for channel in tuple_)


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
        return cls(
            red=rgba.red,
            green=rgba.green,
            blue=rgba.blue,
            alpha=rgba.alpha,
        )

    def _to(self) -> 'RGBA':
        return self

    def _to_tuple(self, *args, **kwargs) -> tuple[float, ...]:
        return super()._to_tuple(*args, **kwargs) + (self._alpha, )
