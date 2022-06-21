from abc import abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F

from .base import LOSSES, BaseLoss


class FunctionalLoss(BaseLoss):

    @staticmethod
    @abstractmethod
    def func(*args, **kwargs) -> torch.Tensor:
        pass

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if mask is None:
            loss = self.func(
                pred,
                target,
                *args,
                reduction=self.reduction,
                **kwargs,
            )
        else:
            loss = self.func(pred, target, *args, reduction='none', **kwargs)
            loss = self.reduce(loss, mask)
        return loss


# TODO: move to mimic.py
class _2DMixin(FunctionalLoss):

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        _, _, h, w = pred.shape
        if pred.shape != target.shape:
            target = F.adaptive_avg_pool2d(target, (h, w))
        if mask is not None and pred.shape != mask.shape:
            mask = F.adaptive_avg_pool2d(mask, (h, w))
        return super().forward(pred, target, mask, *args, **kwargs)


class NormMixin(FunctionalLoss):

    def __init__(self, *args, norm: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._norm = norm

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self._norm:
            pred = F.normalize(pred)
            target = F.normalize(target)
        return super().forward(pred, target, *args, **kwargs)


@LOSSES.register_module()
class L1Loss(NormMixin, FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.l1_loss(*args, **kwargs)


@LOSSES.register_module()
class L12DLoss(_2DMixin, L1Loss):
    pass


@LOSSES.register_module()
class MSELoss(NormMixin, FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.mse_loss(*args, **kwargs)


@LOSSES.register_module()
class MSE2DLoss(_2DMixin, MSELoss):
    pass


@LOSSES.register_module()
class BCELoss(FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy(*args, **kwargs)


@LOSSES.register_module()
class BCEWithLogitsLoss(FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(*args, **kwargs)


@LOSSES.register_module()
class CrossEntropyLoss(FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.cross_entropy(*args, **kwargs)
