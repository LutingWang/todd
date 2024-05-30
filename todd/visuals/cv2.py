__all__ = [
    'CV2Visual',
]

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from ..colors import BGR, RGB, Color
from ..configs import Config
from ..registries import VisualRegistry
from .anchors import XAnchor, YAnchor
from .raster import RasterVisual

Image = npt.NDArray[np.uint8]


@VisualRegistry.register_()
class CV2Visual(RasterVisual):

    def __init__(
        self,
        width: int,
        height: int,
        channels: int = 3,
        **kwargs,
    ) -> None:
        self._image = np.zeros(
            (height, width, channels),
            dtype=np.uint8,
            **kwargs,
        )

    @property
    def width(self) -> int:
        return self._image.shape[1]

    @property
    def height(self) -> int:
        return self._image.shape[0]

    def save(self, path: Any) -> None:
        cv2.imwrite(path, self._image)

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

        h, w, _ = image.shape
        if width is not None or height is not None:
            w, h = self._scale_wh((w, h), width, height)
            image = cv2.resize(image, (w, h))

        self._image[top:top + h, left:left + w] *= 1 - opacity
        self._image[top:top + h, left:left + w] += image * opacity

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

    def text(
        self,
        text: str,
        x: int,
        y: int,
        x_anchor: XAnchor = XAnchor.LEFT,
        y_anchor: YAnchor = YAnchor.TOP,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
        font: Config | None = None,
        thickness: int = 1,
    ) -> Image:
        if font is None:
            font = Config()

        font_face = font.get('face', cv2.FONT_HERSHEY_COMPLEX_SMALL)
        font_scale = font.get('scale', 1.0)

        wh, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        xy = self._translate_xy(wh, x, y, x_anchor, y_anchor)

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
