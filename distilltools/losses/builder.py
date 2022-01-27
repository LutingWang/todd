from typing import Dict, List, Optional

from mmcv.utils import Registry
import torch

from ..adapts import AdaptLayer, AdaptModuleList


LOSSES = Registry('losses')


class LossModuleList(AdaptModuleList):
    def __init__(self, *args, layer_kwargs: Optional[dict] = None, **kwargs):
        layer_kwargs = {} if layer_kwargs is None else layer_kwargs
        layer_kwargs['registry'] = LOSSES
        super().__init__(*args, layer_kwargs=layer_kwargs, **kwargs)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        losses = super().forward(*args, inplace=False, **kwargs)
        losses = {f'loss_{k}': v for k, v in losses.items()}
        return losses
