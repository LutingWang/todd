from typing import Any

import pytest
import torch

from todd_tasks.object_detection import FlattenBBoxesXYWH
from todd_tasks.object_detection.datasets.coco import (
    Annotation as COCOAnnotation,
)
from todd_tasks.object_detection.datasets.coco import (
    Annotations as COCOAnnotations,
)
from todd_tasks.object_detection.datasets.lvis import (
    Annotation as LVISAnnotation,
)
from todd_tasks.object_detection.datasets.lvis import (
    Annotations as LVISAnnotations,
)
from todd_tasks.object_detection.datasets.objects365 import (
    Annotation as Objects365Annotation,
)
from todd_tasks.object_detection.datasets.objects365 import (
    Annotations as Objects365Annotations,
)


@pytest.mark.parametrize(
    ('annotations_type', 'annotation'),
    [
        (
            COCOAnnotations,
            COCOAnnotation(torch.zeros(1), 2., False, (1., 2., 3., 4.), 0),
        ),
        (
            LVISAnnotations,
            LVISAnnotation(2., torch.zeros(1), (1., 2., 3., 4.), 0),
        ),
        (
            Objects365Annotations,
            Objects365Annotation(
                2.,
                (1., 2., 3., 4.),
                0,
                False,
                False,
                False,
            ),
        ),
    ],
)
@pytest.mark.parametrize('empty', [False, True])
def test_bboxes(
    annotations_type: type,
    annotation: Any,
    empty: bool,
) -> None:
    annotations = annotations_type([] if empty else [annotation])

    bboxes = annotations.bboxes

    assert isinstance(bboxes, FlattenBBoxesXYWH)
    assert bboxes.shape == ((0, ) if empty else (1, ))
