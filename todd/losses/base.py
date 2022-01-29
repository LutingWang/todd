from typing_extensions import Literal

from mmcv.runner import BaseModule
import torch


class BaseLoss(BaseModule):
    def __init__(
        self, reduction: Literal['none', 'mean', 'sum'] ='mean', 
        weight: float = 1.0, **kwargs,
    ):
        super().__init__(**kwargs)
        self.reduction = reduction
        self._weight = weight

    def forward(self, loss: torch.Tensor) -> torch.Tensor:
        return self._weight * loss
