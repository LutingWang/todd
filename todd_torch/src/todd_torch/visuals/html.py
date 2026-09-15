__all__ = [
    'HTMLVisual',
]

import math
import shutil
from collections.abc import Sequence
from contextlib import contextmanager
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator

import numpy as np
import numpy.typing as npt
from jinja2 import Template
from playwright.sync_api import Browser, Page

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
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        assert not list(path.iterdir())

        resources_root = Path(__file__).with_suffix('')
        shutil.copytree(resources_root, path, dirs_exist_ok=True)
        for template_path in path.rglob('*.jinja'):
            template: Template = Template(template_path.read_text())
            template_path.with_suffix('').write_text(
                template.render(
                    width=self.width,
                    height=self.height,
                    elements=self._elements,
                ),
            )

    @contextmanager
    def _export(self, browser: Browser) -> Generator[Page, None, None]:
        with TemporaryDirectory() as temporary_root:
            html_root = Path(temporary_root)
            self.save(html_root)
            html_path = html_root / 'index.html'
            page = browser.new_page(
                viewport=dict(
                    width=self.width,
                    height=self.height,
                ),
            )
            try:
                page.goto(html_path.as_uri())
                if error := page.evaluate('window.todd_render()'):
                    raise ValueError(error)
                yield page
            finally:
                page.close()

    def export_pdf(self, *args, path: Path, **kwargs) -> None:
        with self._export(*args, **kwargs) as page:
            page.pdf(
                path=path,
                print_background=True,
                prefer_css_page_size=True,
            )

    def export_screenshot(self, *args, path: Path, **kwargs) -> None:
        with self._export(*args, **kwargs) as page:
            page.screenshot(path=path)

    @staticmethod
    def _to_rgb(color: Color) -> RGB:
        if not isinstance(color, RGB):
            color = color.to(RGB)
        return color

    def _resolve_wh(
        self,
        position: Point,
        width: float | None,
        height: float | None,
    ) -> tuple[float, float]:
        if width is None:
            width = self.width - position.x
        if height is None:
            height = self.height - position.y
        return width, height

    def point(
        self,
        point: Point,
        pen: Pen,
    ) -> str:
        color = self._to_rgb(pen.color)
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
        color = self._to_rgb(pen.color)
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
        color = self._to_rgb(color)
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
        color = self._to_rgb(style.color)
        width, height = self._resolve_wh(
            position,
            width,
            height,
        )
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

    def latex(
        self,
        latex: str,
        position: Point,
        style: TextStyle,
        width: float | None = None,
        height: float | None = None,
        display_mode: bool = False,
    ) -> str:
        color = self._to_rgb(style.color)
        width, height = self._resolve_wh(
            position,
            width,
            height,
        )
        self._elements.append(
            '<div class="todd-element todd-semantic todd-latex" '
            f'data-todd-display-mode="{"true" if display_mode else "false"}" '
            f'style="left:{position.x:g}px;top:{position.y:g}px;'
            f'width:{width:g}px;height:{height:g}px;'
            f'font-size:{style.font_size:g}px;'
            f'color:{color.to_css()}">{escape(latex)}</div>',
        )
        return self._elements[-1]

    def table(
        self,
        html: str,
        position: Point,
        style: TextStyle,
        width: float | None = None,
        height: float | None = None,
    ) -> str:
        color = self._to_rgb(style.color)
        width, height = self._resolve_wh(
            position,
            width,
            height,
        )
        self._elements.append(
            '<div class="todd-element todd-semantic todd-table" '
            f'style="left:{position.x:g}px;top:{position.y:g}px;'
            f'width:{width:g}px;height:{height:g}px;'
            f'font-size:{style.font_size:g}px;'
            f'color:{color.to_css()}">{html}</div>',
        )
        return self._elements[-1]
