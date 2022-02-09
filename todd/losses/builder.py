from typing import Dict

from mmcv.utils import Registry
import torch

from ..adapts import AdaptLayer, AdaptModuleList


LOSSES = Registry('losses')


class LossLayer(AdaptLayer):
    REGISTRY = LOSSES


class LossModuleList(AdaptModuleList):
    LAYER_TYPE = LossLayer

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        losses = super().forward(*args, inplace=False, **kwargs)
        losses = {f'loss_{k}': v for k, v in losses.items()}
        return losses
