from typing import Optional, Tuple

import torch


def clamp(x: torch.Tensor, min_: float = 0) -> torch.Tensor:
    if not x.is_cuda and x.dtype is torch.float16:
        x = x.float().clamp_min(min_).half()
    else:
        x = x.clamp_min(0)
    return x


def iou(bboxes1: torch.Tensor, bboxes2: Optional[torch.Tensor] = None, eps: float = 1e-6):
    """
    Args:
        bboxes1: *1 x 4
        bboxes2: *2 x 4
    
    Returns:
        ious: *1 x *2
    """
    flag = bboxes2 is None
    if flag:
        bboxes2 = bboxes1
    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return bboxes1.new_empty((bboxes1.shape[0], bboxes2.shape[0]))

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (
        area1 if flag else 
        (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    )

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
    union = clamp(union, eps)
    ious = intersection / union
    return ious


def expand_bboxes(bboxes: torch.Tensor, ratio: float, image_shape: Optional[Tuple[int]] = None) -> torch.Tensor:
    """
    Args:
        bboxes: * x 4
    
    Returns:
        expanded_bboxes: * x 4
    """
    bboxes = bboxes.clone()

    bboxes[..., :2] -= bboxes[..., 2:] * (ratio - 1) / 2
    bboxes[..., :2].clamp_min_(0)

    bboxes[..., 2:] *= ratio
    if image_shape is not None:
        h, w = image_shape
        bboxes[..., 2] = torch.min(bboxes[..., 2], w - bboxes[..., 0])
        bboxes[..., 3] = torch.min(bboxes[..., 3], h - bboxes[..., 1])

        # bboxes[..., 2:] += bboxes[..., :2]
        # bboxes[..., 2].clamp_max_(w)
        # bboxes[..., 3].clamp_max_(h)
        # bboxes[..., 2:] -= bboxes[..., :2]

    return bboxes
