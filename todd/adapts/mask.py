from typing import List, Tuple
import einops

import torch

from .base import BaseAdapt
from .builder import ADAPTS


@ADAPTS.register_module()
class Mask(BaseAdapt):
    def __init__(self, *args, strides: List[int], **kwargs):
        super().__init__(*args, **kwargs)
        self._strides = strides

    def _instance(self, mask: torch.Tensor, bboxes: torch.Tensor):
        """
        Args:
            mask: h x w
            bboxes: m x 4

        Returns:
            mask: h x w
        """
        values = 1.0 / (bboxes[:, 2:] - bboxes[:, :2] + 2).prod(1)
        values = values.float()
        for i in range(values.numel()):
            area = mask[bboxes[i, 1]:bboxes[i, 3] + 2, bboxes[i, 0]:bboxes[i, 2] + 2]
            torch.maximum(area, values[i], out=area)
        return mask

    def _batch(self, feat: torch.Tensor, bboxes: List[torch.Tensor]):
        """
        Args:
            feat: n x c x h x w
            bboxes: n x m x 4

        Returns:
            masks: n x h x w
            bg_masks: n x h x w
        """
        n, _, h, w = feat.shape
        assert len(bboxes) == n
        masks = feat.new_zeros((n, h, w))
        for i in range(n):
            self._instance(masks[i], bboxes[i].int())
        masks = einops.rearrange(masks, 'n h w -> n 1 h w')
        bg_masks = masks <= 0
        bg_values = einops.reduce(bg_masks, 'n 1 h w -> n 1 1 1', reduction='sum').clamp_min_(1)
        bg_masks = bg_masks / bg_values
        return masks, bg_masks
    
    def forward(self, bboxes: List[torch.Tensor], feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            feats: l x n x c x h x w
            bboxes: n x m x 4

        Returns:
            masks: l x n x h x w
            bg_masks: l x n x h x w
        """
        assert len(self._strides) == len(feats)
        masks, bg_masks = zip(*[
            self._batch(feat, [b / stride for b in bboxes]) 
            for feat, stride in zip(feats, self._strides)
        ])
        return masks, bg_masks
