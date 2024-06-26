__all__ = [
    'BaseVisual',
]

from abc import ABC, abstractmethod
from typing import Any, Iterable, Self

import cv2
import numpy as np
import numpy.typing as npt
import torch

from ..bases.configs import Config
from ..colors import PALETTE, RGB, Color
from ..patches.cv2 import ColorMap


class BaseVisual(ABC):

    @abstractmethod
    def __init__(self, width: int, height: int) -> None:
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    def color(self, index: int) -> Color:
        index %= len(PALETTE)
        return PALETTE[index]

    @abstractmethod
    def save(self, path: Any) -> None:
        pass

    @abstractmethod
    def image(
        self,
        image: npt.NDArray[np.uint8],
        left: int = 0,
        top: int = 0,
        width: int | None = None,
        height: int | None = None,
        opacity: float = 1.0,
    ) -> Any:
        pass

    def activation(
        self,
        activation: torch.Tensor,
        left: int = 0,
        top: int = 0,
        width: int | None = None,
        height: int | None = None,
        opacity: float = 0.5,
    ) -> Any:
        """Draw the activation map.

        Suppose our activation map is :math:`(256, 13, 20)`, where 256 is the
        number of channels, 13 is the height, and 20 is the width:

            >>> activation = torch.rand(256, 13, 20)

        We first reduce the channel dimension, using whatever reduction:

            >>> import einops
            >>> activation = einops.reduce(
            ...     activation,
            ...     'c h w -> h w',
            ...     reduction='mean',
            ... )

        Then we draw the activation map:

            >>> from .pptx import PPTXVisual
            >>> visual = PPTXVisual(640, 426)
            >>> visual.activation(
            ...     activation,
            ...     width=visual.width,
            ...     height=visual.height,
            ... )
            <pptx.shapes.picture.Picture object at ...>

        Args:
            activation: :math:`(H, W)`
            left: x coordinate of the left side of the activation map
            top: y coordinate of the top size of the activation map
            width: width of the activation map
            height: height of the activation map
            opacity: opacity of the activation map
        """
        color_map = ColorMap(cv2.COLORMAP_JET)
        image = color_map(activation.detach())
        return self.image(image, left, top, width, height, opacity)

    @abstractmethod
    def rectangle(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
        thickness: int = 1,
        fill: Color | None = None,
    ) -> Any:
        pass

    @abstractmethod
    def text(
        self,
        text: str,
        x: int,
        y: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
        font: Config | None = None,
    ) -> Any:
        pass

    @abstractmethod
    def point(
        self,
        x: int,
        y: int,
        size: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
    ) -> Any:
        pass

    @abstractmethod
    def marker(
        self,
        x: int,
        y: int,
        size: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
    ) -> Any:
        pass

    def scatter(
        self,
        points: Iterable[tuple[int, int]],
        sizes: Iterable[int],
        colors: Iterable[Color],
        types: Iterable[str],
    ) -> Any:
        for (x, y), size, color, type_ in zip(points, sizes, colors, types):
            if type_ == '.':
                self.point(x, y, size, color)
            elif type_ == '*':
                self.marker(x, y, size, color)
            else:
                raise ValueError(f'Invalid type: {type_}')
        return self

    @abstractmethod
    def line(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
        thickness: int = 1,
    ) -> Any:
        pass

    def trajectory(
        self,
        trajectory: Iterable[tuple[int, int]],
        color: Color = RGB(0., 0., 0.),  # noqa: B008
        thickness: int = 1,
    ) -> Self:
        """Draw the trajectory.

        Args:
            trajectory: :math:`(T, 2)`
            color: color of the trajectory
            thickness: thickness of the trajectory
        """
        trajectory = list(trajectory)
        for (x1, y1), (x2, y2) in zip(trajectory[:-1], trajectory[1:]):
            self.line(x1, y1, x2, y2, color, thickness)
        return self
