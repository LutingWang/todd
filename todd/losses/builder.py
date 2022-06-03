from typing import Dict

import torch
from mmcv.utils import Registry

from ..adapts import AdaptLayer, AdaptModuleList

LOSSES = Registry('losses')


class LossLayer(AdaptLayer):
    REGISTRY = LOSSES


class LossModuleList(AdaptModuleList):
    LAYER_TYPE = LossLayer

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        # yapf: disable
        losses = super().forward(  # type: ignore[misc]
            *args, inplace=False, **kwargs,
        )
        # yapf: enable
        losses = {f'loss_{k}': v for k, v in losses.items()}
        return losses
