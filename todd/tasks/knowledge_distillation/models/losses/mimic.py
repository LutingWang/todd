__all__ = [
    'L12DLoss',
    'MSE2DLoss',
]

from abc import ABC

import torch
import torch.nn.functional as F

from todd.models.losses import FunctionalLoss, L1Loss, MSELoss

from ..registries import KDLossRegistry


class _2DMixin(FunctionalLoss, ABC):  # noqa: N801

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        _, _, h, w = pred.shape
        if pred.shape != target.shape:
            target = F.adaptive_avg_pool2d(target, (h, w))
        if mask is not None and pred.shape != mask.shape:
            mask = F.adaptive_avg_pool2d(mask, (h, w))
        return super().forward(pred, target, *args, mask=mask, **kwargs)


@KDLossRegistry.register_()
class L12DLoss(_2DMixin, L1Loss):
    pass


@KDLossRegistry.register_()
class MSE2DLoss(_2DMixin, MSELoss):
    pass
