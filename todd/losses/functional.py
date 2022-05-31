from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .base import BaseLoss
from .builder import LOSSES


# TODO: move to mimic.py
class _2DMixin(BaseLoss):
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        *args, **kwargs,
    ) -> torch.Tensor:
        _, _, h, w = pred.shape
        if pred.shape != target.shape:
            target = F.adaptive_avg_pool2d(target, (h, w))
        if mask is not None and pred.shape != mask.shape:
            mask = F.adaptive_avg_pool2d(mask, (h, w))
        return super().forward(pred, target, mask, *args, **kwargs)


class NormMixin(BaseLoss):
    def __init__(self, *args, norm: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._norm = norm

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        *args, **kwargs,
    ) -> torch.Tensor:
        if self._norm:
            pred = F.normalize(pred)
            target = F.normalize(target)
        return super().forward(pred, target, *args, **kwargs)


class FunctionalLoss(BaseLoss):
    func: Callable[..., torch.Tensor]

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        *args, **kwargs,
    ) -> torch.Tensor:
        if mask is None:
            loss = self.func(pred, target, reduction=self.reduction)
        else:
            loss = self.func(pred, target, reduction='none')
            loss = self.reduce(loss, mask)
        return super().forward(loss, *args, **kwargs)


@LOSSES.register_module()
class L1Loss(NormMixin, FunctionalLoss):
    func = staticmethod(F.l1_loss)


@LOSSES.register_module()
class L12DLoss(_2DMixin, L1Loss):
    pass


@LOSSES.register_module()
class MSELoss(NormMixin, FunctionalLoss):
    func = staticmethod(F.mse_loss)


@LOSSES.register_module()
class MSE2DLoss(_2DMixin, MSELoss):
    pass


@LOSSES.register_module()
class BCELoss(FunctionalLoss):
    func = staticmethod(F.binary_cross_entropy)


@LOSSES.register_module()
class BCEWithLogitsLoss(FunctionalLoss):
    func = staticmethod(F.binary_cross_entropy_with_logits)
