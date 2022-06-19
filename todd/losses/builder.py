from typing import Dict

import torch
from mmcv.utils import Registry

from ..base import ModuleJob, ModuleStep

LOSSES = Registry('losses')


class LossLayer(ModuleStep):
    REGISTRY = LOSSES


class LossModuleList(ModuleJob):
    STEP_TYPE = LossLayer

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        losses = super().forward(  # type: ignore[misc]
            *args, inplace=False, **kwargs,
        )
        losses = {f'loss_{k}': v for k, v in losses.items()}
        return losses
