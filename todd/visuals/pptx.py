__all__ = [
    'PPTXVisual',
]

import io

import cv2
import numpy as np
import numpy.typing as npt
import pptx
import pptx.dml.color
import pptx.enum.shapes
import pptx.enum.text
import pptx.parts.image
import pptx.presentation
import pptx.shapes.autoshape
import pptx.shapes.picture
import pptx.shapes.shapetree
import pptx.slide
import pptx.util

from ..base import VisualRegistry
from .base import BaseVisual, Color, XAnchor, YAnchor


@VisualRegistry.register_()
class PPTXVisual(BaseVisual):
    """Visualize data in the format of PowerPoint.

    The PowerPoint contains only one slide.
    For more details, refer to python-pptx_.

    .. _python-pptx: https://github.com/scanny/python-pptx
    """

    def __init__(self, width: int, height: int) -> None:
        """Initializes the PowerPoint with a single slide.

        To initialize a PowerPoint with width 640pt and height 426pt, use the
        following code::

            >>> visual = PPTXVisual(640, 426)

        Once initialized, the width and height of the slide cannot be altered.
        We can read the width and height of the PowerPoint by::

            >>> visual.width
            640
            >>> visual.height
            426

        Args:
            width: the width of the PowerPoint in point
            height: the height of the PowerPoint in point
        """
        super().__init__(width, height)

        pre: pptx.presentation.Presentation = pptx.Presentation()
        pre.slide_width = pptx.util.Pt(width)
        pre.slide_height = pptx.util.Pt(height)

        layout: pptx.slide.SlideLayout = pre.slide_layouts[6]
        slides: pptx.slide.Slides = pre.slides
        slide: pptx.slide.Slide = slides.add_slide(layout)
        shapes: pptx.shapes.shapetree.SlideShapes = slide.shapes

        self._pre = pre
        self._shapes = shapes

    @property
    def width(self) -> int:
        """Width of the PowerPoint."""
        width: pptx.util.Pt = self._pre.slide_width
        return int(width.pt)

    @property
    def height(self) -> int:
        """Height of the PowerPoint."""
        height: pptx.util.Pt = self._pre.slide_height
        return int(height.pt)

    def save(self, path) -> None:
        """Save the PowerPoint.

        The save target can either be a filepath, for example::

            >>> import tempfile
            >>> with tempfile.NamedTemporaryFile() as f:
            ...     PPTXVisual(640, 426).save(f.name)

        Or it can simply be a file-like object::

            >>> with tempfile.TemporaryFile() as f:
            ...     PPTXVisual(640, 426).save(f)

        Args:
            path: destination path
        """
        self._pre.save(path)

    def image(
        self,
        image: npt.NDArray[np.uint8],
        left: int = 0,
        top: int = 0,
        width: int | None = None,
        height: int | None = None,
        opacity: float = 1.0,
    ) -> pptx.shapes.picture.Picture:
        """Add an image to the PowerPoint.

        Suppose the image is :math:`(426, 640)`::

            >>> image = np.random.randint(0, 256, (426, 640, 3))

        In most cases, the PowerPoint is of the same size as the image, so
        that the image covers the whole background::

            >>> h, w, _ = image.shape
            >>> visual = PPTXVisual(w, h)
            >>> visual.image(image)
            <pptx.shapes.picture.Picture object at ...>

        The returned `pptx.shapes.picture.Picture` object can be used to
        fine-tune the properties of the image.
        Note that the DPI of the image should not be changed thoughtlessly,
        because of the bizarre measurements in PowerPoint.
        The most common measurement unit in PowerPoint is *Point* (pt), where
        1pt equals 1/72 inches.
        However, images are measured in pixels and 1 pixel equals 1/DPI inches.
        By default, DPI is 72 so that 1 pixel is 1pt.
        When DPI is set to other values, the size of the image may become
        larger or smaller than it is supposed to be.

        Args:
            image: :math:`(H, W, 3)`
            left: x coordinate of the left side of the image
            top: y coordinate of the top side of the image
            width: width of the image
            height: height of the image
            opacity: opacity of the image
        """
        h, w, c = image.shape
        assert c == 3

        left_pixels = pptx.util.Pt(left)
        top_pixels = pptx.util.Pt(top)
        width_pixels = width if width is None else pptx.util.Pt(width)
        height_pixels = height if height is None else pptx.util.Pt(height)

        assert 0.0 <= opacity <= 1.0

        alpha = np.ones((h, w, 1)) * 255 * opacity
        image = np.concatenate([image, alpha], axis=-1)
        status, image = cv2.imencode('.png', image)
        assert status
        bytes_io = io.BytesIO(image.tobytes())
        picture = self._shapes.add_picture(
            bytes_io,
            left_pixels,
            top_pixels,
            width_pixels,
            height_pixels,
        )

        image_part: pptx.parts.image.Image = picture.image
        assert image_part.dpi == (72, 72)

        return picture

    def rectangle(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        color: Color = Color(0, 0, 0),
    ) -> pptx.shapes.autoshape.Shape:
        autoshape_type = pptx.enum.shapes.MSO_AUTO_SHAPE_TYPE
        rectangle: pptx.shapes.autoshape.Shape = self._shapes.add_shape(
            autoshape_type.RECTANGLE,  # pylint: disable=no-member
            pptx.util.Pt(left),
            pptx.util.Pt(top),
            pptx.util.Pt(width),
            pptx.util.Pt(height),
        )

        fill: pptx.shapes.autoshape.FillFormat = rectangle.fill
        fill.background()

        line: pptx.shapes.autoshape.LineFormat = rectangle.line
        line.width = pptx.util.Pt(1)

        line_color: pptx.dml.color.ColorFormat = line.color
        line_color.rgb = pptx.dml.color.RGBColor(*color)

        return rectangle

    def text(
        self,
        text: str,
        x: int,
        y: int,
        x_anchor: XAnchor = XAnchor.LEFT,
        y_anchor: YAnchor = YAnchor.BOTTOM,
        color: Color = Color(0, 0, 0),
    ) -> pptx.shapes.autoshape.Shape:
        width = pptx.util.Pt(9 * len(text))
        height = pptx.util.Pt(15)

        if x_anchor is XAnchor.LEFT:
            left = pptx.util.Pt(x)
        elif x_anchor is XAnchor.RIGHT:
            left = pptx.util.Pt(x) - width
        else:
            raise ValueError(f"Unsupported anchor {x_anchor}")

        if y_anchor is YAnchor.TOP:
            top = pptx.util.Pt(y)
        elif y_anchor is YAnchor.BOTTOM:
            top = pptx.util.Pt(y) - height
        else:
            raise ValueError(f"Unsupported anchor {y_anchor}")

        textbox: pptx.shapes.autoshape.Shape = self._shapes.add_textbox(
            left,
            top,
            width,
            height,
        )

        text_frame = textbox.text_frame
        text_frame.margin_left = 0
        text_frame.margin_top = 0
        text_frame.margin_right = 0
        text_frame.margin_bottom = 0
        text_frame.vertical_anchor = getattr(
            pptx.enum.text.MSO_VERTICAL_ANCHOR,
            y_anchor.name,
        )

        paragraph = text_frame.paragraphs[0]
        paragraph.text = text

        font = paragraph.font
        font.name = 'Times New Roman'
        font.size = pptx.util.Pt(12)

        font_color: pptx.dml.color.ColorFormat = font.color
        font_color.rgb = pptx.dml.color.RGBColor(*color)

        return textbox
