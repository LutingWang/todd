__all__ = [
    'HTMLVisual',
]

import math
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from jinja2 import Template

from todd.colors import RGB, Color

from ..registries import VisualRegistry
from ..utils.networks import image_to_data_url
from .base import BaseVisual, Pen, Point, TextStyle


@VisualRegistry.register_()
class HTMLVisual(BaseVisual):

    def __init__(
        self,
        *args,
        width: int,
        height: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, width=width, height=height, **kwargs)
        self._width = width
        self._height = height
        self._elements: list[str] = []

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def save(self, path: Any) -> None:
        html_root = Path(__file__).with_suffix('')
        html_path = html_root / 'index.html.jinja'
        html = html_path.read_text()
        script_path = html_root / 'script.js'
        script = script_path.read_text()
        template: Template = Template(html)
        document = template.render(
            width=self.width,
            height=self.height,
            elements=''.join(self._elements),
            script=script,
        )
        path_: Path = Path(path)
        path_.write_text(document)

    def point(
        self,
        point: Point,
        pen: Pen,
    ) -> str:
        color = pen.color
        if not isinstance(color, RGB):
            color = color.to(RGB)
        self._elements.append(
            '<div class="todd-element todd-point" '
            f'style="left:{point.x:g}px;top:{point.y:g}px;'
            f'width:{pen.width:g}px;height:{pen.width:g}px;'
            f'background:{color.to_css()}"></div>',
        )
        return self._elements[-1]

    def line(
        self,
        start: Point,
        end: Point,
        pen: Pen,
    ) -> str:
        dx = end.x - start.x
        dy = end.y - start.y
        color = pen.color
        if not isinstance(color, RGB):
            color = color.to(RGB)
        self._elements.append(
            '<div class="todd-element todd-line" '
            f'style="left:{start.x:g}px;top:{start.y - pen.width / 2:g}px;'
            f'width:{math.hypot(dx, dy):g}px;height:{pen.width:g}px;'
            f'background:{color.to_css()};'
            f'transform:rotate({math.degrees(math.atan2(dy, dx)):g}deg)"'
            '></div>',
        )
        return self._elements[-1]

    def fill(
        self,
        points: Sequence[Point],
        color: Color,
    ) -> str:
        points_ = ','.join(f'{point.x:g}px {point.y:g}px' for point in points)
        if not isinstance(color, RGB):
            color = color.to(RGB)
        self._elements.append(
            '<div class="todd-element todd-fill" '
            f'style="background:{color.to_css()};'
            f'clip-path:polygon({points_})"></div>',
        )
        return self._elements[-1]

    def text(
        self,
        text: str,
        position: Point,
        style: TextStyle,
        width: float | None = None,
        height: float | None = None,
    ) -> str:
        color = style.color
        if not isinstance(color, RGB):
            color = color.to(RGB)
        if width is None:
            width = self.width - position.x
        if height is None:
            height = self.height - position.y
        self._elements.append(
            '<div class="todd-element todd-text" data-todd-fit-text '
            f'data-todd-font-size="{style.font_size:g}" '
            f'style="left:{position.x:g}px;top:{position.y:g}px;'
            f'width:{width:g}px;height:{height:g}px;'
            f'font-size:{style.font_size:g}px;'
            f'color:{color.to_css()}">'
            f'{escape(text)}</div>',
        )
        return self._elements[-1]

    def image(
        self,
        image: npt.NDArray[np.uint8],
        position: Point,
        width: float | None = None,
        height: float | None = None,
        opacity: float = 1,
    ) -> str:
        width, height = self._get_image_wh(image, width, height)
        data_url = image_to_data_url(image)
        self._elements.append(
            '<img class="todd-element todd-image" alt="" '
            f'src="{data_url}" '
            f'style="left:{position.x:g}px;top:{position.y:g}px;'
            f'width:{width:g}px;height:{height:g}px;'
            f'opacity:{opacity:g}">',
        )
        return self._elements[-1]
