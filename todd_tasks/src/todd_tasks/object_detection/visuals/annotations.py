__all__ = [
    'annotation',
    'annotations',
]

from typing import Any, Iterable

from todd.colors import RGB, Color
from todd_torch.visuals import BaseVisual, Pen, Point, TextStyle

from ..bboxes import FlattenBBoxesMixin, FlattenBBoxesXYWH


def annotation(
    visual: BaseVisual,
    left: int,
    top: int,
    width: int,
    height: int,
    text: str | None = None,
    color: Color = RGB(red=0., green=0., blue=0.),  # noqa: B008
    thickness: int = 1,
    font_size: float = 12,
) -> tuple[Any, Any]:
    """Draw an annotation bbox.

    Each annotation comprises a bbox and a textual label.
    The bbox is given by (left, top, width, height).
    The text is labeled above the bbox, left-aligned.

    The method is useful to visualize dataset annotations or pseudo
    labels, for example:

        >>> from todd_torch.visuals import PPTXVisual
        >>> visual = PPTXVisual(width=640, height=426)
        >>> annotations = [
        ...     dict(bbox=[236.98, 142.51, 24.7, 69.5], category_id=64),
        ...     dict(bbox=[7.03, 167.76, 149.32, 94.87], category_id=72),
        ... ]
        >>> categories = {64: 'potted plant', 72: 'tv'}
        >>> for a in annotations:
        ...     category_id = a['category_id']
        ...     category_name = categories[category_id]
        ...     rectangle, text = annotation(
        ...         visual,
        ...         *map(int, a['bbox']),
        ...         category_name,
        ...         RGB.from_hex('#dc2626'),
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
    rectangle = visual.rectangle(
        Point(left, top),
        Point(left + width, top + height),
        pen=Pen(color=color, width=thickness),
    )
    text_ = None if text is None else visual.text(
        text,
        Point(left, top + height),
        TextStyle(
            color=color,
            font_size=font_size,
        ),
    )
    return rectangle, text_


def annotations(
    visual: BaseVisual,
    bboxes: FlattenBBoxesMixin,
    texts: Iterable[str | None] | None = None,
    **kwargs,
) -> list[tuple[Any, Any]]:
    bboxes = bboxes.to(FlattenBBoxesXYWH).flatten()
    if texts is None:
        texts = [None] * len(bboxes)
    else:
        texts = list(texts)
        assert len(texts) == len(bboxes)

    return [
        annotation(  # type: ignore[call-arg]
            visual,
            *map(int, bbox),  # type: ignore[arg-type]
            text,
            **kwargs,
        )
        for text, bbox in zip(texts, bboxes)
    ]
