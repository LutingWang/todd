from typing_extensions import Literal

from mmcv.runner import BaseModule
import torch

from .schedualers import SchedualerConfig, build_schedualer


class BaseLoss(BaseModule):
    def __init__(
        self, reduction: Literal['none', 'mean', 'sum'] ='mean', 
        weight: SchedualerConfig = 1.0, **kwargs,
    ):
        super().__init__(**kwargs)
        self.reduction = reduction

        weight = build_schedualer(weight)
        self._weight = weight

    def forward(self, loss: torch.Tensor) -> torch.Tensor:
        return self._weight * loss
