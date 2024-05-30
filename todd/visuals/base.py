__all__ = [
    'BaseVisual',
]

from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from ..colors import PALETTE, RGB, Color
from ..configs import Config
from ..patches.cv2 import ColorMap
from .anchors import XAnchor, YAnchor


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
            activation: :math:`(H, W)` or :math:`(H, W, 1)`
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
        x_anchor: XAnchor = XAnchor.LEFT,
        y_anchor: YAnchor = YAnchor.TOP,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
        font: Config | None = None,
    ) -> Any:
        pass

    def annotation(
        self,
        text: str,
        left: int,
        top: int,
        width: int,
        height: int,
        color: Color = RGB(0., 0., 0.),  # noqa: B008
        thickness: int = 1,
        font: Config | None = None,
    ) -> tuple[Any, Any]:
        """Draw an annotation bbox.

        Each annotation comprises a bbox and a textual label.
        The bbox is given by (left, top, width, height).
        The text is labeled above the bbox, left-aligned.

        The method is useful to visualize dataset annotations or pseudo
        labels, for example:

            >>> from .pptx import PPTXVisual
            >>> visual = PPTXVisual(640, 426)
            >>>
            >>> annotations = [
            ...     dict(bbox=[236.98, 142.51, 24.7, 69.5], category_id=64),
            ...     dict(bbox=[7.03, 167.76, 149.32, 94.87], category_id=72),
            ... ]
            >>> categories = {64: 'potted plant', 72: 'tv'}
            >>> for annotation in annotations:
            ...     category_id = annotation['category_id']
            ...     category_name = categories[category_id]
            ...     rectangle, text = visual.annotation(
            ...         category_name,
            ...         *annotation['bbox'],
            ...         visual.color(category_id),
            ...     )
            >>> rectangle
            <pptx.shapes.autoshape.Shape object at ...>
            >>> text
            <pptx.shapes.autoshape.Shape object at ...>

        Args:
            text: annotated text along with the bbox. Typically is the class
                name or class id.
            left: x coordinate of the bbox
            top: y coordinate of the bbox
            width: width of the bbox
            height: height of the bbox
            color: color of the bbox

        Returns:
            tuple of the bbox and the text object
        """
        rectangle = self.rectangle(
            left,
            top,
            width,
            height,
            color,
            thickness,
        )
        text_ = self.text(
            text,
            left,
            top + height,
            color=color,
            font=font,
        )
        return rectangle, text_
