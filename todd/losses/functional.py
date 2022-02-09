import torch
import torch.nn.functional as F

from .base import BaseLoss
from .builder import LOSSES


# TODO: refactor


@LOSSES.register_module()
class MSELoss(BaseLoss):
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ):
        loss = F.mse_loss(pred, target, reduction='none')
        return super().forward(loss, *args, **kwargs)


@LOSSES.register_module()
class L1Loss(BaseLoss):
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ):
        loss = F.l1_loss(pred, target, reduction='none')
        return super().forward(loss, *args, **kwargs)
