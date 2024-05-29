__all__ = [
    'FocalLoss',
    'FocalWithLogitsLoss',
]

from abc import ABC

import torch

from ....models.losses import BCELoss, BCEWithLogitsLoss, FunctionalLoss
from ....models.registries import LossRegistry


class FocalMixin(FunctionalLoss, ABC):

    def __init__(
        self,
        *args,
        gamma=2.0,
        alpha=0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._gamma = gamma
        self._alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        pt = (1 - pred) * target + pred * (1 - target)
        weight = self._alpha * target + (1 - self._alpha) * (1 - target)
        return super().forward(
            pred,
            target,
            weight * pt.pow(self._gamma),
            *args,
            **kwargs,
        )


@LossRegistry.register_()
class FocalLoss(FocalMixin, BCELoss):
    pass


@LossRegistry.register_()
class FocalWithLogitsLoss(FocalMixin, BCEWithLogitsLoss):
    pass
