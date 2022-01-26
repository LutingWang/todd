from typing import Dict, List

from mmcv.utils import Registry
import torch

from ..adapts import AdaptLayer, AdaptModuleList


LOSSES = Registry('losses')


class LossModuleList(AdaptModuleList):
    def __init__(self, losses: List[dict], **kwargs):
        losses = [
            AdaptLayer.build(loss, registry=LOSSES) 
            for loss in losses
        ]
        super().__init__(losses, **kwargs)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        losses = super().forward(*args, inplace=False, **kwargs)
        losses = {f'loss_{k}': v for k, v in losses.items()}
        return losses
