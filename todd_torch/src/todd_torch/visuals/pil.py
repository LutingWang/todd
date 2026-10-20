__all__ = [
    'PILVisual',
]

import math
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from todd.colors import RGBA, Color

from ..registries import VisualRegistry
from .base import BaseVisual, Pen, Point, TextStyle


@VisualRegistry.register_()
class PILVisual(BaseVisual[Image.Image]):

    @classmethod
    def apply_opacity(
        cls,
        image: Image.Image,
        opacity: float,
    ) -> Image.Image:
        assert 0 <= opacity <= 1
        if opacity == 1:
            return image
        image = image.convert('RGBA')
        alpha = image.getchannel('A').point(
            lambda value: round(value * opacity),
        )
        image.putalpha(alpha)
        return image

    @staticmethod
    def _to_rgba(color: Color) -> tuple[int, int, int, int]:
        r, g, b, a = map(round, RGBA.from_(color).to_tuple(False))
        return r, g, b, a

    def __init__(self, *args, width: int, height: int, **kwargs) -> None:
        super().__init__(*args, width=width, height=height, **kwargs)
        self._canvas = Image.new('RGBA', (width, height))

    @property
    def width(self) -> int:
        return self._canvas.width

    @property
    def height(self) -> int:
        return self._canvas.height

    def to_pil(self) -> Image.Image:
        return self._canvas

    def to_numpy(self) -> npt.NDArray[np.uint8]:
        return np.array(self._canvas)

    def save(self, path: Any) -> None:
        self._canvas.save(path)

    @contextmanager
    def _drawing(self) -> Generator[ImageDraw.ImageDraw]:
        layer = Image.new('RGBA', self._canvas.size)
        yield ImageDraw.Draw(layer)
        self._canvas.alpha_composite(layer)

    def point(self, point: Point, pen: Pen) -> Image.Image:
        x, y = point
        radius = pen.width / 2
        with self._drawing() as draw:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                self._to_rgba(pen.color),
            )
        return self._canvas

    def line(self, start: Point, end: Point, pen: Pen) -> Image.Image:
        with self._drawing() as draw:
            draw.line(
                (start.round_(), end.round_()),
                self._to_rgba(pen.color),
                math.ceil(pen.width),
            )
        return self._canvas

    def fill(self, points: Sequence[Point], color: Color) -> Image.Image:
        with self._drawing() as draw:
            draw.polygon(
                [point.round_() for point in points],
                self._to_rgba(color),
            )
        return self._canvas

    def text(
        self,
        text: str,
        position: Point,
        style: TextStyle,
        width: float | None = None,
        height: float | None = None,
    ) -> Image.Image:
        with self._drawing() as draw:
            draw.text(
                position.round_(),
                text,
                self._to_rgba(style.color),
                ImageFont.load_default(size=math.ceil(style.font_size)),
            )
        return self._canvas

    def image(
        self,
        image: Image.Image,
        position: Point,
        width: float | None = None,
        height: float | None = None,
        opacity: float = 1,
    ) -> Image.Image:
        w, h = self._scale_wh(image.size, width, height)
        image = image.convert('RGBA').resize(
            (math.ceil(w), math.ceil(h)),
            Image.Resampling.LANCZOS,
        )
        image = self.apply_opacity(image, opacity)
        self._canvas.alpha_composite(image, position.round_())
        return self._canvas
