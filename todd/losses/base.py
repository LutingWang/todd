__all__ = [
    'BaseLoss',
]

from typing import Literal

import torch

from ..base import Config
from ..utils import Module
from .schedulers import BaseScheduler, SchedulerRegistry

Reduction = Literal['none', 'mean', 'sum', 'prod']


class BaseLoss(Module):

    def __init__(
        self,
        reduction: Reduction = 'mean',
        weight: float | Config = 1.0,
        bound: float | None = None,
        **kwargs,
    ) -> None:
        if isinstance(weight, float):
            weight = Config(type='ConstantScheduler', gain=weight)
        if bound is not None and bound <= 1e-4:
            raise ValueError('bound must be greater than 1e-4')
        super().__init__(**kwargs)
        self._reduction = reduction
        self._weight: BaseScheduler = SchedulerRegistry.build(weight)
        self._threshold = None if bound is None else bound / self.weight

        self.register_forward_hook(forward_hook)

    @property
    def reduction(self) -> Reduction:
        return self._reduction

    @property
    def weight(self) -> float:
        return self._weight()

    @property
    def threshold(self) -> float | None:
        return self._threshold

    def reduce(
        self,
        loss: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            loss = loss * mask
        if self._reduction == 'none':
            pass
        elif self._reduction in ['sum', 'mean', 'prod']:
            loss = getattr(loss, self._reduction)()
        else:
            raise NotImplementedError(self._reduction)
        return loss


def forward_hook(
    module: BaseLoss,
    input_,
    output: torch.Tensor,
) -> torch.Tensor:
    weight = module.weight
    if module.threshold is None:
        return weight * output

    # coef = bound / (weight * output)
    coef = module.threshold / output.item()
    # if bound < weight * output
    if coef < 1.0:
        # weight = bound / output
        weight *= coef

    output = weight * output
    return output
