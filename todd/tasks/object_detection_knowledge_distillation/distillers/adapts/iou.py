__all__ = [
    'IoU',
]

import torch

from ....knowledge_distillation.distillers import AdaptRegistry
from ....knowledge_distillation.distillers.adapts import BaseAdapt
from ....object_detection import BBoxesXYXY


@AdaptRegistry.register_()
class IoU(BaseAdapt):

    def __init__(
        self,
        *args,
        aligned: bool = False,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert not aligned
        self._aligned = aligned
        self._eps = eps

    def _reshape(
        self,
        bboxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Size]:
        """Reshape BBoxes.

        Args:
            bboxes: * x 4

        Returns:
            bboxes: prod(*) x 4
            shape: tuple(*)
        """
        bboxes = bboxes[..., :4]
        # bboxes = bboxes.half()
        shape = bboxes.shape[:-1]
        bboxes = bboxes.reshape(-1, 4)
        return bboxes, shape

    def _iou(
        self,
        bboxes1: torch.Tensor,
        bboxes2: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute IoU.

        Args:
            bboxes1: \*1 x 4
            bboxes2: \*2 x 4

        Returns:
            ious: \*1 x \*2
        """
        bboxes1, shape1 = self._reshape(bboxes1)
        bboxes2, shape2 = self._reshape(bboxes2)
        ious = BBoxesXYXY.ious(
            BBoxesXYXY(bboxes1),
            BBoxesXYXY(bboxes2),
            self._eps,
        )
        return ious.reshape(shape1 + shape2)

    def forward(
        self,
        bboxes1: list[torch.Tensor],
        bboxes2: list[torch.Tensor],
    ) -> list[list[torch.Tensor]]:
        """Compute IoU.

        Args:
            bboxes1: n1 x m1 x 4
            bboxes2: n2 x m2 x 4

        Returns:
            ious: n1 x n2 x m1 x m2
        """
        return [[self._iou(bbox1, bbox2)
                 for bbox2 in bboxes2]
                for bbox1 in bboxes1]
