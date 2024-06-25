__all__ = [
    'DeFeatMask',
    'FGDMask',
    'FGFIMask',
    'FRSMask',
]

import math
from abc import ABC, abstractmethod

import einops
import torch

import todd.tasks.knowledge_distillation as kd

from ..registries import ODKDAdaptRegistry

BaseAdapt = kd.distillers.adapts.BaseAdapt


class MultiLevelMask:

    def __init__(
        self,
        *args,
        strides: list[int],
        ceil_mode: bool = False,
        **kwargs,
    ) -> None:
        self._strides = strides
        self._ceil_mode = ceil_mode
        super().__init__(*args, **kwargs)

    @property
    def strides(self) -> list[int]:
        return self._strides

    @abstractmethod
    def _forward(
        self,
        shape: tuple[int, int],
        bboxes: list[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        pass

    def _div(self, a: int, b: int) -> int:
        if self._ceil_mode:
            return math.ceil(a / b)
        return a // b

    def forward(
        self,
        shape: tuple[int, int],
        bboxes: list[torch.Tensor],
        *args,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Compute Multilevel Mask.

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
            mask = self._forward(level_shape, level_bboxes, *args, **kwargs)
            masks.append(mask)
        return masks


class SingleLevelMask(MultiLevelMask, ABC):

    def __init__(self, *args, stride: int, **kwargs) -> None:
        super().__init__(*args, strides=[stride], **kwargs)

    @property
    def stride(self) -> int:
        return self.strides[0]

    def forward(  # type: ignore[override]
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        masks = super().forward(*args, **kwargs)
        return masks[0]


@ODKDAdaptRegistry.register_()
class DeFeatMask(MultiLevelMask, BaseAdapt):

    def __init__(
        self,
        *args,
        neg_gain: float = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._neg_gain = neg_gain

    @staticmethod
    def _normalize(masks: torch.Tensor) -> torch.Tensor:
        """Normalize masks.

        Args:
            masks: n x 1 x h x w

        Returns:
            normalized_masks: n x 1 x h x w
        """
        values = einops.reduce(
            masks,
            'n 1 h w -> n 1 1 1',
            reduction='sum',
        ).clamp_min_(1)
        normalized_masks = torch.true_divide(masks, values)
        return normalized_masks

    def _normalize_pos(self, masks: torch.Tensor) -> torch.Tensor:
        return self._normalize(masks)

    def _normalize_neg(self, masks: torch.Tensor) -> torch.Tensor:
        return self._normalize(masks)

    def _mask(
        self,
        shape: tuple[int, int],
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DeFeat Mask.

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

    def _forward(
        self,
        shape: tuple[int, int],
        bboxes: list[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute DeFeat Mask.

        Args:
            shape: (h, w)
            bboxes: n x m x 4

        Returns:
            masks: n x 1 x h x w
        """
        masks = torch.stack([self._mask(shape, bbox) for bbox in bboxes])
        masks = einops.rearrange(masks, 'n h w -> n 1 h w')
        masks = self._normalize_pos(masks)
        neg_masks = self._normalize_neg(masks <= 0)
        neg_masks = neg_masks * self._neg_gain
        return masks + neg_masks


@ODKDAdaptRegistry.register_()
class FGDMask(DeFeatMask):

    def _normalize_pos(self, masks: torch.Tensor) -> torch.Tensor:
        return masks

    def _mask(
        self,
        shape: tuple[int, int],
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute FGD Mask.

        Args:
            shape: (h, w)
            bboxes: m x 4

        Returns:
            mask: h x w
        """
        mask = bboxes.new_zeros(shape)
        bboxes = bboxes.int()
        values = torch.true_divide(
            1.0,
            (bboxes[:, 2:] - bboxes[:, :2] + 2).prod(1),
        )
        for i, (x0, y0, x1, y1) in enumerate(bboxes.tolist()):
            area = mask[y0:y1 + 2, x0:x1 + 2]
            torch.maximum(area, values[i], out=area)
        return mask


@ODKDAdaptRegistry.register_()
class FGFIMask(BaseAdapt):

    def __init__(self, *args, thresh: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._thresh = thresh

    def _instance(self, ious: torch.Tensor) -> torch.Tensor:
        """Compute FGFI Mask for an instance.

        Args:
            ious: h x w x k x m

        Returns:
            mask: h x w
        """
        thresh = einops.reduce(
            ious,
            'h w k m -> 1 1 1 m',
            reduction='max',
        ) * self._thresh
        mask = einops.reduce(ious > thresh, 'h w k m -> h w', reduction='max')
        return mask

    def _batch(self, ious: list[torch.Tensor]) -> torch.Tensor:
        """Compute FGFI Mask for a batch.

        Args:
            ious: n x h x w x k x m

        Returns:
            masks: n x 1 x h x w
        """
        masks = torch.stack([self._instance(iou) for iou in ious])
        masks = einops.rearrange(masks, 'n h w -> n 1 h w')
        return masks

    def forward(self, ious: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        """Compute FGFI Mask.

        Args:
            ious: l x n x h x w x k x m

        Returns:
            masks: l x n x 1 x h x w
        """
        masks = [self._batch(iou) for iou in ious]
        return masks


# @ADAPTS.register()
# class DenseCLIPMask(SingleLevelMask, LabelEncMask):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, stride=32, aug=False, **kwargs)

#     def _mask(
#         self,
#         shape: tuple[int, int],
#         bboxes: torch.Tensor,
#         labels: torch.Tensor,
#     ) -> torch.Tensor:
#         masks = bboxes.new_zeros(self._num_classes, *shape)
#         for (x0, y0, x1, y1), label in zip(
#             bboxes.int().tolist(),
#             labels.tolist(),
#         ):
#             masks[label, y0:y1 + 1, x0:x1 + 1] = 1
#         return masks


@ODKDAdaptRegistry.register_()
class FRSMask(BaseAdapt):

    def __init__(self, *args, with_logits: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._with_logits = with_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute FRS Mask.

        Args:
            x: n x c x h x w

        Returns:
            masks: n x 1 x h x w
        """
        x = x.detach()
        masks = einops.reduce(x, 'n c h w -> n 1 h w', reduction='max')
        if self._with_logits:
            masks = masks.sigmoid()
        return masks
