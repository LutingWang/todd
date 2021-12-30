from typing_extensions import Literal

import torch
import torch.nn.functional as F

from .base import BaseLoss
from .builder import LOSSES
from .utils import weight_loss


@weight_loss
def mse_loss(pred: torch.Tensor, target: torch.Tensor):
    return F.mse_loss(pred, target, reduction='none')


@LOSSES.register_module()
class MSELoss(BaseLoss):
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None,
        reduction_override: Literal['none', 'mean', 'sum'] = None,
    ):
        reduction = reduction_override or self.reduction
        loss = mse_loss(pred, target, weight=weight, reduction=reduction)
        return super().forward(loss)
