import math
from typing import List, Optional, Tuple, Union

import einops
import torch

from .base import BaseAdapt
from .builder import ADAPTS


@ADAPTS.register_module()
class DeFeatMask(BaseAdapt):
    def __init__(
        self, 
        *args, 
        neg_gain: float = 4, 
        strides: Union[int, List[int]], 
        ceil_mode: bool = False, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._neg_gain = neg_gain
        self._strides = strides
        self._ceil_mode = ceil_mode

    def _normalize(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masks: n x 1 x h x w

        Returns:
            normalized_masks: n x 1 x h x w
        """
        values = einops.reduce(masks, 'n 1 h w -> n 1 1 1', reduction='sum').clamp_min_(1)
        normalized_masks = torch.true_divide(masks, values)
        return normalized_masks

    def _normalize_pos(self, masks: torch.Tensor) -> torch.Tensor:
        return self._normalize(masks)

    def _normalize_neg(self, masks: torch.Tensor) -> torch.Tensor:
        return self._normalize(masks)

    def _mask(
        self, 
        shape: Tuple[int, int], 
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            shape: (h, w)
            bboxes: m x 4

        Returns:
            mask: h x w
        """
        mask = bboxes.new_zeros(shape)
        for x0, y0, x1, y1 in bboxes.int().tolist():
            mask[y0:y1 + 2, x0:x1 + 2] = 1
        return mask

    def _pos(
        self, 
        shape: Tuple[int, int], 
        bboxes: List[torch.Tensor], 
    ) -> torch.Tensor:
        """
        Args:
            shape: (h, w)
            bboxes: n x m x 4

        Returns:
            masks: n x 1 x h x w
        """
        masks = [self._mask(shape, bbox) for bbox in bboxes]
        masks = torch.stack(masks)
        masks = einops.rearrange(masks, 'n h w -> n 1 h w')
        return self._normalize_pos(masks)

    def _neg(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masks: n x 1 x h x w

        Returns:
            neg_masks: n x 1 x h x w
        """
        neg_masks = self._normalize_neg(masks <= 0)
        return neg_masks * self._neg_gain

    def _forward(
        self, 
        shape: Tuple[int, int], 
        bboxes: List[torch.Tensor], 
        stride: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            shape: (h, w)
            bboxes: n x m x 4

        Returns:
            masks: n x 1 x h x w
            neg_masks: n x 1 x h x w
        """
        if stride is None:
            assert isinstance(self._strides, int)
            stride = self._strides
        h, w = shape
        if self._ceil_mode:
            shape = (math.ceil(h / stride), math.ceil(w / stride))
        else:
            shape = (h // stride, w // stride)
        bboxes = [bbox / stride for bbox in bboxes]
        masks = self._pos(shape, bboxes)
        neg_masks = self._neg(masks)
        return masks + neg_masks

    def forward(
        self, 
        shape: Tuple[int, int], 
        bboxes: List[torch.Tensor],
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            shape: (h, w)
            bboxes: n x m x 4

        Returns:
            masks: l x n x 1 x h x w
            neg_masks: l x n x 1 x h x w
        """
        if isinstance(self._strides, list):
            masks = [
                self._forward(shape, bboxes, stride) 
                for stride in self._strides
            ]
        elif isinstance(self._strides, int):
            masks = self._forward(shape, bboxes)
        else:
            assert False
        return masks


@ADAPTS.register_module()
class FGDMask(DeFeatMask):
    def _normalize_pos(self, masks: torch.Tensor) -> torch.Tensor:
        return masks

    def _mask(
        self, 
        shape: Tuple[int, int], 
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
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
        for i, (x0, y0, x1, y1) in enumerate(bboxes.tolist()):
            area = mask[y0:y1 + 2, x0:x1 + 2]
            torch.maximum(area, values[i], out=area)
        return mask


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
