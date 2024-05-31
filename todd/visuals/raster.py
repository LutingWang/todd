__all__ = [
    'RasterVisual',
]

from abc import ABC

from .base import BaseVisual, XAnchor, YAnchor


class RasterVisual(BaseVisual, ABC):

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
