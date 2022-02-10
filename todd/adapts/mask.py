from typing import List, Optional, Tuple
import einops

import torch

from .base import BaseAdapt
from .builder import ADAPTS


@ADAPTS.register_module()
class FGDMask(BaseAdapt):
    def __init__(self, *args, strides: List[int], **kwargs):
        super().__init__(*args, **kwargs)
        self._strides = strides

    def _instance(self, shape: Tuple[int, int], bboxes: torch.Tensor):
        """
        Args:
            shape: (h, w)
            bboxes: m x 4

        Returns:
            mask: h x w
        """
        mask = bboxes.new_zeros(shape)
        bboxes = bboxes.int()
        values = torch.true_divide(1.0, (bboxes[:, 2:] - bboxes[:, :2] + 2).prod(1))
        for i in range(values.numel()):
            area = mask[bboxes[i, 1]:bboxes[i, 3] + 2, bboxes[i, 0]:bboxes[i, 2] + 2]
            torch.maximum(area, values[i], out=area)
        return mask

    def _batch(self, shape: Tuple[int, int], bboxes: List[torch.Tensor], stride: Optional[int] = None):
        """
        Args:
            shape: (h, w)
            bboxes: n x m x 4

        Returns:
            masks: n x 1 x h x w
            bg_masks: n x 1 x h x w
        """
        if stride is None:
            assert self._strides is None
            stride = self._strides
        h, w = shape
        shape = (h // stride, w // stride)
        bboxes = [bbox / stride for bbox in bboxes]
        masks = [self._instance(shape, bbox) for bbox in bboxes]
        masks = torch.stack(masks)
        masks = einops.rearrange(masks, 'n h w -> n 1 h w')
        bg_masks = masks <= 0
        bg_values = einops.reduce(bg_masks, 'n 1 h w -> n 1 1 1', reduction='sum').clamp_min_(1)
        bg_masks = torch.true_divide(bg_masks, bg_values)
        return masks, bg_masks
    
    def forward(self, shape: Tuple[int, int], bboxes: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            shape: (h, w)
            bboxes: n x m x 4

        Returns:
            masks: l x n x 1 x h x w
            bg_masks: l x n x 1 x h x w
        """
        if isinstance(self._strides, list):
            masks, bg_masks = zip(*[
                self._batch(shape, bboxes, stride) 
                for stride in self._strides
            ])
        elif isinstance(self._strides, int):
            masks, bg_masks = self._batch(shape, bboxes)
        else:
            assert False
        return masks, bg_masks


@ADAPTS.register_module()
class FGFIMask(BaseAdapt):
    def __init__(self, *args, thresh: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._thresh = thresh

    def _instance(self, ious: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ious: h x w x k x m

        Returns:
            mask: h x w
        """
        thresh = einops.reduce(ious, 'h w k m -> 1 1 1 m', reduction='max') * self._thresh
        mask = einops.reduce(ious >= thresh, 'h w k m -> h w', reduction='max')
        return mask

    def _batch(self, ious: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            ious: n x h x w x k x m

        Returns:
            masks: n x 1 x h x w
        """
        masks = [self._instance(iou) for iou in ious]
        masks = torch.stack(masks)
        masks = einops.rearrange(masks, 'n h w -> n 1 h w')
        return masks

    def forward(self, ious: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Args:
            ious: l x n x h x w x k x m

        Returns:
            masks: l x n x 1 x h x w
        """
        masks = [self._batch(iou) for iou in ious]
        return masks
