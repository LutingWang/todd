__all__ = [
    'FunctionalLoss',
    'NormMixin',
    'L1Loss',
    'MSELoss',
    'BCELoss',
    'BCEWithLogitsLoss',
    'CrossEntropyLoss',
    'CosineEmbeddingLoss',
]

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from ..base import LossRegistry
from .base import BaseLoss


class FunctionalLoss(BaseLoss):

    @staticmethod
    @abstractmethod
    def func(*args, **kwargs) -> torch.Tensor:
        pass

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args,
        mask: torch.Tensor | None = None,
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


class NormMixin(FunctionalLoss, ABC):

    def __init__(self, *args, norm: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._norm = norm

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self._norm:
            pred = F.normalize(pred)
            target = F.normalize(target)
        return super().forward(pred, target, *args, mask=mask, **kwargs)


@LossRegistry.register_()
class L1Loss(NormMixin, FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.l1_loss(*args, **kwargs)


@LossRegistry.register_()
class MSELoss(NormMixin, FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.mse_loss(*args, **kwargs)


@LossRegistry.register_()
class BCELoss(FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy(*args, **kwargs)


@LossRegistry.register_()
class BCEWithLogitsLoss(FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(*args, **kwargs)


@LossRegistry.register_()
class CrossEntropyLoss(FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.cross_entropy(*args, **kwargs)


@LossRegistry.register_()
class CosineEmbeddingLoss(FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.cosine_embedding_loss(*args, **kwargs)
