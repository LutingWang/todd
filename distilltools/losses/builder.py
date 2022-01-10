from typing import Dict

from mmcv.cnn import MODELS
from mmcv.utils import Registry
import torch

from ..adapts import AdaptLayer, AdaptModuleDict


LOSSES = Registry('losses', parent=MODELS)


class LossModuleDict(AdaptModuleDict):
    def __init__(self, losses: dict, **kwargs):
        losses = {
            k: AdaptLayer.build(v, registry=LOSSES) 
            for k, v in losses.items()
        }
        super().__init__(losses, **kwargs)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        losses = super().forward(*args, inplace=False, **kwargs)
        losses = {f'loss_{k}': v for k, v in losses.items()}
        return losses
