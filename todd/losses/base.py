from typing import Literal

from mmcv.runner import BaseModule
import torch
from torch.nn._reduction import get_enum


class BaseLoss(BaseModule):
    def __init__(
        self, reduction: Literal['none', 'mean', 'sum'] ='mean', 
        weight: float = 1.0, **kwargs,
    ):
        super().__init__(**kwargs)
        self._reduction = get_enum(reduction)  # none: 0, elementwise_mean: 1, sum: 2
        self._weight = weight

    def forward(self, loss: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            loss = loss * mask
        if self._reduction == 0:
            pass
        elif self._reduction == 1:
            loss = loss.mean()
        elif self._reduction == 2:
            loss = loss.sum()
        else:
            raise Exception
        return self._weight * loss
