__all__ = [
    'CV2Visual',
]

from typing import cast, Optional

import cv2
import numpy as np
import numpy.typing as npt

from .base import BaseVisual, Color, VisualRegistry, XAnchor, YAnchor


@VisualRegistry.register()
class CV2Visual(BaseVisual):

    def __init__(self, width: int, height: int) -> None:
        self._image = np.zeros((height, width, 3))

    @property
    def width(self) -> int:
        return self._image.shape[1]

    @property
    def height(self) -> int:
        return self._image.shape[0]

    def save(self, path) -> None:
        cv2.imwrite(path, self._image)

    def image(
        self,
        image: npt.NDArray[np.uint8],
        left: int = 0,
        top: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        opacity: float = 1.0,
    ) -> npt.NDArray[np.uint8]:
        h, w, c = image.shape
        assert c == 3

        if width is not None or height is not None:
            if width is None:
                width = round(w / h * cast(int, height))
            if height is None:
                height = round(h / w * cast(int, width))
            image = cv2.resize(image, (width, height))
            h, w, _ = image.shape

        assert 0.0 <= opacity <= 1.0
        self._image[top:top + h, left:left + w] *= 1 - opacity
        self._image[top:top + h, left:left + w] += image * opacity

        return self._image.astype(np.uint8)

    def rectangle(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        color: Color = Color(0, 0, 0),
    ) -> npt.NDArray[np.uint8]:
        cv2.rectangle(
            self._image,
            (left, top),
            (left + width, top + height),
            color,
            thickness=1,
        )
        return self._image.astype(np.uint8)

    def text(
        self,
        text: str,
        x: int,
        y: int,
        x_anchor: XAnchor = XAnchor.LEFT,
        y_anchor: YAnchor = YAnchor.BOTTOM,
        color: Color = Color(0, 0, 0),
    ) -> npt.NDArray[np.uint8]:
        assert x_anchor is XAnchor.LEFT
        assert y_anchor is YAnchor.BOTTOM
        cv2.putText(
            self._image,
            text=text,
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            fontScale=1.0,
            color=color,
        )
        return self._image.astype(np.uint8)
