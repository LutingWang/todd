__all__ = [
    'HTMLVisual',
]

import math
import shutil
import tarfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

from jinja2 import Template
from PIL import Image
from playwright.sync_api import Browser, Page

from todd.colors import RGB, Color
from todd.utils import get_bytes

from ..registries import VisualRegistry
from ..utils.networks import image_to_data_url
from .base import BaseVisual, Pen, Point, TextStyle


@VisualRegistry.register_()
class HTMLVisual(BaseVisual[Image.Image]):

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

    def _save_katex(self, input_dir: Path, output_dir: Path) -> None:
        katex_dir = input_dir / '.cache' / 'katex'
        if not katex_dir.exists():
            url = 'https://registry.npmjs.org/katex/-/katex-0.19.0.tgz'
            with (
                get_bytes(url) as f,
                tarfile.open(fileobj=f, mode='r:gz') as archive,
            ):
                archive.extractall(katex_dir, filter='data')
        shutil.copytree(katex_dir / 'package' / 'dist', output_dir / 'katex')

    def _save_script(self, input_dir: Path, output_dir: Path) -> None:
        shutil.copy(input_dir / 'script.js', output_dir / 'script.js')

    def _save_style(self, input_dir: Path, output_dir: Path) -> None:
        template_file = input_dir / 'style.css.jinja'
        template: Template = Template(template_file.read_text())
        output_file = output_dir / 'style.css'
        output_file.write_text(
            template.render(width=self.width, height=self.height),
        )

    def _save_index(self, input_dir: Path, directory: Path) -> None:
        output_dir = directory / 'assets'
        javascript_files = [
            file.relative_to(directory).as_posix() for file in sorted(
                output_dir.rglob('*.js'),
                # Load KaTeX before its plugins.
                key=lambda file: len(str(file)),
            )
        ]
        css_files = [
            file.relative_to(directory).as_posix()
            for file in output_dir.rglob('*.css')
        ]

        template_file = input_dir / 'index.html.jinja'
        template: Template = Template(template_file.read_text())
        output_file = directory / 'index.html'
        output_file.write_text(
            template.render(
                elements=self._elements,
                javascript_files=javascript_files,
                css_files=css_files,
            ),
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        assert not list(path.iterdir())

        input_dir = Path(__file__).with_suffix('')
        output_dir = path / 'assets'
        self._save_katex(input_dir, output_dir)
        self._save_script(input_dir, output_dir)
        self._save_style(input_dir, output_dir)
        self._save_index(input_dir, path)

    @contextmanager
    def open(self, browser: Browser) -> Generator[Page]:
        with TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            self.save(directory)
            file = directory / 'index.html'
            page = browser.new_page(
                viewport=dict(width=self.width, height=self.height),
            )
            try:
                page.goto(file.as_uri())
                if error := page.evaluate('window.todd_render()'):
                    raise ValueError(error)
                yield page
            finally:
                page.close()

    def export_pdf(self, browser: Browser, file: Path) -> None:
        with self.open(browser) as page:
            page.pdf(
                path=file,
                print_background=True,
                prefer_css_page_size=True,
            )

    def export_screenshot(self, browser: Browser, file: Path) -> None:
        with self.open(browser) as page:
            page.screenshot(path=file)

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
        image: Image.Image,
        position: Point,
        width: float | None = None,
        height: float | None = None,
        opacity: float = 1,
    ) -> str:
        width, height = self._scale_wh(image.size, width, height)
        data_url = image_to_data_url(image, '.png')
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
