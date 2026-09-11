__all__ = [
    'CV2Visual',
]

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from todd.colors import RGB, Color

from ..colors import ColorMap
from ..registries import VisualRegistry
from .base import BaseVisual, Pen, Point, TextStyle

Canvas = npt.NDArray[np.uint8]


@VisualRegistry.register_()
class CV2Visual(BaseVisual):

    @staticmethod
    def _to_rgb(color: Color) -> tuple[float, float, float]:
        red, green, blue, *_ = RGB.from_(color).to_tuple(normalized=False)
        return red, green, blue

    def __init__(
        self,
        *args,
        width: int,
        height: int,
        channels: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, width=width, height=height, **kwargs)
        self._canvas: Canvas = np.zeros(
            (height, width, channels),
            dtype=np.uint8,
        )

    @property
    def width(self) -> int:
        return self._canvas.shape[1]

    @property
    def height(self) -> int:
        return self._canvas.shape[0]

    def to_numpy(self) -> Canvas:
        return self._canvas.copy()

    def save(self, path: Any) -> None:
        cv2.imwrite(path, cv2.cvtColor(self._canvas, cv2.COLOR_RGB2BGR))

    def point(
        self,
        point: Point,
        pen: Pen,
        *,
        marker_type: int = cv2.MARKER_CROSS,
    ) -> Canvas:
        cv2.drawMarker(
            self._canvas,
            point.round_(),
            self._to_rgb(pen.color),
            marker_type,
            round(pen.width),
            1,
            cv2.LINE_AA,
        )
        return self._canvas

    def line(
        self,
        start: Point,
        end: Point,
        pen: Pen,
    ) -> Canvas:
        cv2.line(
            self._canvas,
            start.round_(),
            end.round_(),
            self._to_rgb(pen.color),
            round(pen.width),
            cv2.LINE_AA,
        )
        return self._canvas

    def fill(
        self,
        points: Sequence[Point],
        color: Color,
    ) -> Canvas:
        points_ = np.array(
            [p.round_() for p in points],
            dtype=np.int32,
        )
        cv2.fillPoly(
            self._canvas,
            [points_],
            self._to_rgb(color),
        )
        return self._canvas

    def text(
        self,
        text: str,
        position: Point,
        style: TextStyle,
        width: float | None = None,
        height: float | None = None,
    ) -> Canvas:
        (_, font_height), baseline = (
            cv2.getTextSize('Ag', cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
        )
        scale = style.font_size / (font_height + baseline)
        cv2.putText(
            self._canvas,
            text,
            (
                round(position.x),
                round(position.y + font_height * scale),
            ),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            scale,
            self._to_rgb(style.color),
            1,
            cv2.LINE_AA,
        )
        return self._canvas

    def image(
        self,
        image: Canvas,
        position: Point,
        width: float | None = None,
        height: float | None = None,
        opacity: float = 1,
    ) -> Canvas:
        assert 0 <= opacity <= 1

        w, h = self._get_image_wh(image, width, height)
        w, h = round(w), round(h)
        if (h, w) != image.shape[:2]:
            image = cv2.resize(image, (w, h))  # type: ignore[assignment]

        x, y = position.round_()
        background = self._canvas[y:y + h, x:x + w]
        cv2.addWeighted(
            background,
            1 - opacity,
            image,
            opacity,
            0,
            background,
        )
        return self._canvas

    def heatmap(
        self,
        values: torch.Tensor,
        position: Point = Point(0, 0),  # noqa: B008
        width: float | None = None,
        height: float | None = None,
        opacity: float = .5,
        *,
        color_map: int = cv2.COLORMAP_JET,
    ) -> Canvas:
        image = ColorMap(color_map=color_map)(values.detach().cpu())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.image(image, position, width, height, opacity)
