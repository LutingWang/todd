import torch
import torch.nn.functional as F

from .base import BaseLoss
from .builder import LOSSES
from .functional import MSELoss


@LOSSES.register_module()
class MimicLoss(MSELoss):
    def __init__(self, norm: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor, *args, **kwargs):
        target = F.adaptive_avg_pool2d(target, pred.shape[-2:])
        if self.norm:
            pred = F.normalize(pred)
            target = F.normalize(target)
        return super().forward(pred, target, *args, **kwargs)


@LOSSES.register_module()
class FGFILoss(MSELoss):
    def forward(
        self, 
        pred: torch.Tensor, target: torch.Tensor, 
        mask: torch.Tensor, 
        *args, **kwargs,
    ) -> torch.Tensor:
        assert mask.dtype == torch.bool
        return super().forward(pred[mask], target[mask], *args, **kwargs)


@LOSSES.register_module()
class DeFeatLoss(BaseLoss):
    def __init__(self, *args, neg_gain: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._neg_gain = neg_gain

    def forward(
        self, 
        pred: torch.Tensor, target: torch.Tensor, 
        mask: torch.Tensor, neg_mask: torch.Tensor, 
        *args, **kwargs,
    ) -> torch.Tensor:
        _, _, h, w = pred.shape
        target = F.adaptive_avg_pool2d(target, (h, w))
        loss = F.mse_loss(pred, target, reduction='none')
        pos = loss * mask
        neg = loss * neg_mask * self._neg_gain
        loss = self.reduce(pos + neg)
        return super().forward(loss, *args, **kwargs)


@LOSSES.register_module()
class DevLoss(MimicLoss):
    def __init__(self, pos_share: float = 1, *args, **kwargs):
        assert 0 <= pos_share <= 1
        super().__init__(*args, **kwargs)
        self.pos_share = pos_share

    def pos_mask(self, mask: torch.Tensor) -> torch.Tensor:
        pos = mask.sum().item()
        total = mask.numel()
        pos_frac = self.pos_share * total / pos if pos > 0 else 0
        neg_frac = (1 - self.pos_share)  * total / (total - pos) if pos < total else 0
        weight = torch.zeros_like(mask, dtype=torch.float)
        weight.masked_fill_(mask, pos_frac)
        weight.masked_fill_(~mask, neg_frac)
        return weight
    
    def iou_mask(self, mask: torch.Tensor, pos_thresh: float = 0.5, index: float = 2) -> torch.Tensor:
        """Adjust IoU mask to satisfy `pos_share` requirement.

        Adjust IoU mask to retrieve `weight` w such that
        $$
            w = a * mask^index + b, \\quad a, b \\in \\R^+
            w_p = w * (w > pos_thresh)
            sum(w) = mask.numel()
            sum(w_p) = sum(w) * self.pos_share
        $$

        The equations are sometimes unsatisfiable. The algorithms goes briefly as follows:
        1. estimate b' to balance pos and neg
        2. normalize sum of weight to `mask.numel()`
        """
        assert mask.requires_grad == False
        n_total = mask.numel()
        pos_mask = mask > pos_thresh
        n_pos = pos_mask.sum()
        mask = mask ** index
        total = mask.sum()
        pos = mask[pos_mask].sum()
        base = (pos / self.pos_share - total) / (n_total - n_pos / self.pos_share)
        base.clamp_(1e-5, 1)
        mask += base
        weight = mask * n_total / mask.sum()
        return weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        mask = mask.unsqueeze(1)  # bs x 1 x h x w
        if mask.dtype == torch.bool:
            mask = self.pos_mask(mask)
        elif mask.dtype == torch.float:
            mask = self.iou_mask(mask)
        else:
            mask = torch.ones_like(pred)

        if mask.shape != pred.shape:
            mask = F.interpolate(mask, size=pred.shape[-2:])

        return super().forward(pred, target, *args, mask=mask, **kwargs)
