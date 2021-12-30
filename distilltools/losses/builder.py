from typing import Dict

from mmcv.cnn import MODELS
from mmcv.utils import Registry
import torch

from ..adapts import AdaptModule


LOSSES = Registry('losses', parent=MODELS)


class LossModule(AdaptModule):
    @classmethod
    def build_adapt(cls, cfg: dict, registry: Registry = LOSSES, adapt_key: str = 'loss'):
        super().build_adapt(cfg, registry, adapt_key)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        losses = super().forward(*args, inplace=False, **kwargs)
        losses = {f'loss_{k}': v for k, v in losses.items()}
