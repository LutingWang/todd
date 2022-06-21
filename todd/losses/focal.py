import torch

from .base import LOSSES
from .functional import BCELoss, BCEWithLogitsLoss, FunctionalLoss


class FocalMixin(FunctionalLoss):

    def __init__(
        self,
        *args,
        gamma=2.0,
        alpha=0.25,
        **kwargs,
    ):
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


@LOSSES.register_module()
class FocalLoss(FocalMixin, BCELoss):
    pass


@LOSSES.register_module()
class FocalWithLogitsLoss(FocalMixin, BCEWithLogitsLoss):
    pass
