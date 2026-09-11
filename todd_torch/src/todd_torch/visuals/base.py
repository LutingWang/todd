__all__ = [
    'PALETTE',
    'Point',
    'Pen',
    'TextStyle',
    'BaseVisual',
]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from todd.colors import HTML4, RGB, Color

PALETTE = tuple(RGB.from_hex(color.value) for color in HTML4)


class Point(NamedTuple):
    x: float
    y: float

    def round_(self) -> tuple[int, int]:
        return round(self.x), round(self.y)


@dataclass(frozen=True)
class Pen:
    color: Color
    width: float


@dataclass(frozen=True)
class TextStyle:
    color: Color
    font_size: float


class BaseVisual(ABC):

    @abstractmethod
    def __init__(
        self,
        *args,
        width: int,
        height: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @abstractmethod
    def save(self, path: Any) -> None:
        pass

    @abstractmethod
    def point(
        self,
        point: Point,
        pen: Pen,
    ) -> Any:
        pass

    @abstractmethod
    def line(
        self,
        start: Point,
        end: Point,
        pen: Pen,
    ) -> Any:
        pass

    def fill(
        self,
        points: Sequence[Point],
        color: Color,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def text(
        self,
        text: str,
        position: Point,
        style: TextStyle,
        width: float | None = None,
        height: float | None = None,
    ) -> Any:
        pass

    @staticmethod
    def _get_image_wh(
        image: npt.NDArray[np.uint8],
        width: float | None,
        height: float | None,
    ) -> tuple[float, float]:
        if width is not None and height is not None:
            return width, height

        h, w = image.shape[:2]

        if width is not None:
            return width, h / w * width
        if height is not None:
            return w / h * height, height
        return w, h

    @abstractmethod
    def image(
        self,
        image: npt.NDArray[np.uint8],
        position: Point,
        width: float | None = None,
        height: float | None = None,
        opacity: float = 1,
    ) -> Any:
        pass

    def color(self, index: int) -> Color:
        return PALETTE[index % len(PALETTE)]

    def polyline(
        self,
        points: Sequence[Point],
        pen: Pen,
    ) -> Any:
        points = tuple(points)
        for start, end in zip(points[:-1], points[1:], strict=True):
            self.line(start, end, pen)
        return self

    def polygon(
        self,
        points: Sequence[Point],
        fill: Color | None = None,
        pen: Pen | None = None,
    ) -> Any:
        points = tuple(points)
        if fill is not None:
            self.fill(points, fill)
        if pen is not None:
            self.polyline((*points, points[0]), pen)
        return self

    def rectangle(
        self,
        left_top: Point,
        right_bottom: Point,
        fill: Color | None = None,
        pen: Pen | None = None,
    ) -> Any:
        left, top = left_top
        right, bottom = right_bottom
        return self.polygon(
            (
                Point(left, top),
                Point(right, top),
                Point(right, bottom),
                Point(left, bottom),
            ),
            fill,
            pen,
        )
