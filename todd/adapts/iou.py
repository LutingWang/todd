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

    def forward(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bboxes1: n x m1 x 4
            bboxes2: n x m2 x 4
        
        Returns:
            ious: n x m1 x m2
        """
        bboxes1 = bboxes1.half()
        bboxes2 = bboxes2.half()
        assert bboxes1.shape[0] == bboxes2.shape[0]
        n = bboxes1.shape[0]
        m1 = bboxes1.shape[1]
        m2 = bboxes2.shape[1]

        if m1 * m2 == 0:
            return bboxes1.new_empty((n, m1, m2))

        area1 = (bboxes1[:, :, 2] - bboxes1[:, :, 0]) * (bboxes1[:, :, 3] - bboxes1[:, :, 1])
        area2 = (bboxes2[:, :, 2] - bboxes2[:, :, 0]) * (bboxes2[:, :, 3] - bboxes2[:, :, 1])

        lt = torch.maximum(  # [n, m1, m2, 2]
            bboxes1[:, :, None, :2],
            bboxes2[:, None, :, :2],
        )
        rb = torch.minimum(  # [n, m1, m2, 2]
            bboxes1[:, :, None, 2:],
            bboxes2[:, None, :, 2:],
        )

        wh = clamp(rb - lt)
        intersection = wh[..., 0] * wh[..., 1]

        union = area1[:, :, None] + area2[:, None, :] - intersection
        union = clamp(union, self._eps)
        ious = intersection / union
        return ious
