__all__ = [
    'CV2Visual',
]

from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from ..bases.configs import Config
from ..colors import BGR, RGB, Color
from ..registries import VisualRegistry
from .anchors import XAnchor, YAnchor
from .base import BaseVisual

Image = npt.NDArray[np.uint8]


@VisualRegistry.register_()
class CV2Visual(BaseVisual):

    def __init__(
        self,
        width: int,
        height: int,
        channels: int = 3,
        **kwargs,
    ) -> None:
        self._image = np.zeros((height, width, channels), **kwargs)

    @property
    def width(self) -> int:
        return self._image.shape[1]

    @property
    def height(self) -> int:
        return self._image.shape[0]

    def to_numpy(self) -> Image:
        return self._image.astype(np.uint8)

    def save(self, path: Any) -> None:
        cv2.imwrite(path, self.to_numpy())

    def _scale_wh(
        self,
        image_wh: tuple[int, int],
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[int, int]:
        w, h = image_wh
        if width is None:
            assert height is not None
            width = round(w / h * height)
        elif height is None:
            height = round(h / w * width)
        return width, height

    def image(
        self,
        image: Image,
        left: int = 0,
        top: int = 0,
        width: int | None = None,
        height: int | None = None,
        opacity: float = 1.0,
    ) -> Image:
        assert 0.0 <= opacity <= 1.0
        image_ = image.astype(np.float32)

        h, w, _ = image_.shape
        if width is not None or height is not None:
            w, h = self._scale_wh((w, h), width, height)
            image_ = cv2.resize(image_, (w, h))

        self._image[top:top + h, left:left + w] *= 1 - opacity
        self._image[top:top + h, left:left + w] += image_ * opacity

        return self._image

    def rectangle(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
        thickness: int = 1,
        fill: Color | None = None,
    ) -> Image:
        args = (
            self._image,
            (left, top),
            (left + width, top + height),
        )
        if fill is not None:
            cv2.rectangle(*args, fill.to(BGR).to_tuple(), thickness=-1)
        cv2.rectangle(*args, color.to(BGR).to_tuple(), thickness=thickness)
        return self._image

    def _translate_xy(
        self,
        text_wh: tuple[int, int],
        x: int,
        y: int,
        x_anchor: XAnchor = XAnchor.LEFT,
        y_anchor: YAnchor = YAnchor.TOP,
    ) -> tuple[int, int]:
        if x_anchor == XAnchor.RIGHT:
            x -= text_wh[0]
        else:
            assert x_anchor is XAnchor.LEFT
        if y_anchor == YAnchor.TOP:
            y += text_wh[1]
        else:
            assert y_anchor is YAnchor.BOTTOM
        return x, y

    def text(
        self,
        text: str,
        x: int,
        y: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
        font: Config | None = None,
        x_anchor: XAnchor = XAnchor.LEFT,
        y_anchor: YAnchor = YAnchor.TOP,
        thickness: int = 1,
    ) -> Image:
        if font is None:
            font = Config()

        font_face = font.get('face', cv2.FONT_HERSHEY_COMPLEX_SMALL)
        font_scale = font.get('scale', 1.0)

        wh, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        xy = self._translate_xy(
            cast(tuple[int, int], wh),
            x,
            y,
            x_anchor,
            y_anchor,
        )

        cv2.putText(
            self._image,
            text,
            xy,
            font_face,
            font_scale,
            color.to(BGR).to_tuple(),
            thickness,
        )
        return self._image

    def point(
        self,
        x: int,
        y: int,
        size: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
    ) -> Image:
        cv2.circle(
            self._image,
            (x, y),
            size,
            color.to(BGR).to_tuple(),
            -1,
            cv2.LINE_AA,
        )
        return self._image

    def marker(
        self,
        x: int,
        y: int,
        size: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
    ) -> Image:
        cv2.drawMarker(
            self._image,
            (x, y),
            color.to(BGR).to_tuple(),
            cv2.MARKER_CROSS,
            5 * size,
            size // 2,
            cv2.LINE_AA,
        )
        return self._image

    def line(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
        thickness: int = 1,
    ) -> Image:
        cv2.line(
            self._image,
            (x1, y1),
            (x2, y2),
            color.to(BGR).to_tuple(),
            thickness,
            cv2.LINE_AA,
        )
        return self._image
