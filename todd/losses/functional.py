import functools
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .base import BaseLoss
from .builder import LOSSES


def wrap(func: Callable[..., torch.Tensor]):

    def wrapper(wrapped_cls: type):
    
        @functools.wraps(wrapped_cls, updated=())
        class WrapperClass(BaseLoss):
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
                    loss = func(pred, target, reduction=self.reduction)
                else:
                    loss = func(pred, target, reduction='none')
                    loss = self.reduce(loss, mask)
                return super().forward(loss, *args, **kwargs)
        
        LOSSES.register_module(module=WrapperClass)
        return WrapperClass

    return wrapper


@wrap(F.l1_loss)
class L1Loss: pass


@wrap(F.mse_loss)
class MSELoss: pass
