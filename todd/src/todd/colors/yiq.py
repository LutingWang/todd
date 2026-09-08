__all__ = [
    'YIQ',
]

from colorsys import rgb_to_yiq, yiq_to_rgb
from typing_extensions import Self

from .color import Color
from .rgba import RGBA


class YIQ(Color):

    def __init__(
        self,
        luminance: float,
        in_phase: float,
        quadrature: float,
    ) -> None:
        assert 0. <= luminance <= 1.
        self._luminance = luminance
        self._in_phase = in_phase
        self._quadrature = quadrature

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
        y, i, q = rgb_to_yiq(rgba.red, rgba.green, rgba.blue)
        return cls(y, i, q)

    def _to(self) -> RGBA:
        r, g, b = yiq_to_rgb(*self.to_tuple())
        return RGBA(r, g, b, alpha=1.)

    def to_tuple(self) -> tuple[float, ...]:
        return self._luminance, self._in_phase, self._quadrature
