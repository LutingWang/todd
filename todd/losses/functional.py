from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .base import BaseLoss
from .builder import LOSSES


class FunctionalLoss(BaseLoss):
    func: Callable[..., torch.Tensor]

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        *args, **kwargs,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            _, _, h, w = pred.shape
            target = F.adaptive_avg_pool2d(target, (h, w))
        if mask is None:
            loss = self.func(pred, target, reduction=self.reduction)
        else:
            loss = self.func(pred, target, reduction='none')
            loss = self.reduce(loss, mask)
        return super().forward(loss, *args, **kwargs)


@LOSSES.register_module()
class L1Loss(FunctionalLoss):
    func = staticmethod(F.l1_loss)


@LOSSES.register_module()
class MSELoss(FunctionalLoss):
    func = staticmethod(F.mse_loss)


@LOSSES.register_module()
class BCEWithLogitsLoss(FunctionalLoss):
    func = staticmethod(F.binary_cross_entropy_with_logits)
