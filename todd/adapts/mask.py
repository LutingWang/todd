import functools
import math
from typing import List, Optional, Tuple, Union

import einops
import torch

from .base import BaseAdapt
from .builder import ADAPTS


def multilevel(wrapped_cls: type):

    @functools.wraps(wrapped_cls, updated=())
    class WrapperClass(wrapped_cls):
        def __init__(
            self,
            *args, 
            strides: List[int],
            ceil_mode: bool = False,
            **kwargs,
        ):
            self._strides = strides
            self._ceil_mode = ceil_mode
            if ceil_mode:
                self._div = lambda a, b: math.ceil(a / b)
            else:
                self._div = lambda a, b: a // b
            super().__init__(*args, **kwargs)

        def forward(
            self, 
            shape: Tuple[int, int], 
            bboxes: List[torch.Tensor],
            *args, **kwargs,
        ) -> List[torch.Tensor]:
            """
            Args:
                shape: (h, w)
                bboxes: n x m x 4
    
            Returns:
                masks: l x n x 1 x h x w
            """
            h, w = shape
            masks = []
            for stride in self._strides:
                level_shape = (self._div(h, stride), self._div(w, stride))
                level_bboxes = [bbox / stride for bbox in bboxes]
                mask = super().forward(level_shape, level_bboxes, *args, **kwargs) 
                masks.append(mask)
            return masks
    
    return WrapperClass


@ADAPTS.register_module()
class LabelEncMask(BaseAdapt):
    def __init__(
        self, 
        *args, 
        num_classes: int = 80, 
        aug: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._num_classes = num_classes
        self._aug = aug

    def _mask(
        self, 
        shape: Tuple[int, int], 
        bboxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            shape: (h, w)
            bboxes: m x 4
            labels: m

        Returns:
            mask: k x h x w
        """
        h, w = shape
        masks = bboxes.new_zeros(self._num_classes, *shape)
        bboxes = torch.cat([
            bboxes[:, 2:] - bboxes[:, :2], 
            (bboxes[:, :2] + bboxes[:, 2:]) / 2,
        ], dim=-1)
        y, x = torch.meshgrid(
            torch.arange(0, shape[0], dtype=torch.float, device=bboxes.device),
            torch.arange(0, shape[1], dtype=torch.float, device=bboxes.device),
        )
        for (w, h, cx, cy), label in zip(bboxes.tolist(), labels.tolist()):
            value = torch.max(
                torch.abs(x - cx) / w, 
                torch.abs(y - cy) / h,
            )
            value = (1 - value) * (value < 0.5)
            if self._aug:
                weight = torch.rand((), device=value.device).clamp_max(0.5) * 2
                value = value * weight
            torch.maximum(masks[label], value, out=masks[label])
        return masks

    def forward(
        self, 
        shape: Tuple[int, int], 
        bboxes: List[torch.Tensor], 
        labels: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            shape: (h, w)
            bboxes: n x m x 4
            labels: n x m

        Returns:
            masks: n x k x h x w
        """
        masks = [self._mask(shape, bbox, label) for bbox, label in zip(bboxes, labels)]
        masks = torch.stack(masks)
        return masks


@ADAPTS.register_module()
@multilevel
class DeFeatMask(BaseAdapt):
    def __init__(
        self, 
        *args, 
        neg_gain: float = 4, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._neg_gain = neg_gain

    @staticmethod
    def _normalize(masks: torch.Tensor) -> torch.Tensor:
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

    def forward(
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
        masks = self._normalize_pos(masks)
        neg_masks = self._normalize_neg(masks <= 0)
        neg_masks = neg_masks * self._neg_gain
        return masks + neg_masks


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
        mask = einops.reduce(ious > thresh, 'h w k m -> h w', reduction='max')
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
