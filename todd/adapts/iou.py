from typing import List, Tuple

import torch

from .base import BaseAdapt
from .builder import ADAPTS


def clamp(x: torch.Tensor, min_: float = 0) -> torch.Tensor:
    if not x.is_cuda and x.dtype is torch.float16:
        x = x.float().clamp_min(min_).half()
    else:
        x = x.clamp_min(0)
    return x


@ADAPTS.register_module()
class IoU(BaseAdapt):
    def __init__(self, *args, aligned: bool = False, eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        assert not aligned
        self._aligned = aligned
        self._eps = eps

    def _reshape(self, bboxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        """
        Args:
            bboxes: * x 4
        
        Returns:
            bboxes: prod(*) x 4
            shape: tuple(*)
        """
        bboxes = bboxes[..., :4].half()
        shape = bboxes.shape[:-1]
        bboxes = bboxes.reshape(-1, 4)
        return bboxes, shape

    def _iou(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bboxes1: *1 x 4
            bboxes2: *2 x 4
        
        Returns:
            ious: *1 x *2
        """
        bboxes1, shape1 = self._reshape(bboxes1)
        bboxes2, shape2 = self._reshape(bboxes2)

        if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
            return bboxes1.new_empty((bboxes1.shape[0], bboxes2.shape[0]))

        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        lt = torch.maximum(  # [*1, *2, 2]
            bboxes1[:, None, :2],
            bboxes2[None, :, :2],
        )
        rb = torch.minimum(  # [*1, *2, 2]
            bboxes1[:, None, 2:],
            bboxes2[None, :, 2:],
        )

        wh = clamp(rb - lt)
        intersection = wh[..., 0] * wh[..., 1]

        union = area1[:, None] + area2[None, :] - intersection
        union = clamp(union, self._eps)
        ious = intersection / union
        return ious.reshape(shape1 + shape2)
    
    def forward(
        self, 
        bboxes1: List[torch.Tensor], 
        bboxes2: List[torch.Tensor],
    ) -> List[List[torch.Tensor]]:
        """
        Args:
            bboxes1: n1 x m1 x 4
            bboxes2: n2 x m2 x 4
        
        Returns:
            ious: n1 x n2 x m1 x m2
        """
        return [
            [self._iou(bbox1, bbox2) for bbox2 in bboxes2]
            for bbox1 in bboxes1
        ]
