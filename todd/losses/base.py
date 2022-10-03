__all__ = [
    'BaseLoss',
    'LOSSES',
]

import numbers
from typing import Any, Literal, Mapping, Optional, Union, cast

import torch
import torch.nn as nn

from ..base import STEPS, Module, Registry
from ..schedulers import SCHEDULERS

Reduction = Literal['none', 'mean', 'sum', 'prod']


class BaseLoss(Module):

    def __init__(
        self,
        reduction: Reduction = 'mean',
        weight: Union[numbers.Real, Mapping] = 1.0,  # type: ignore[assignment]
        bound: Optional[numbers.Real] = None,
        **kwargs,
    ):
        if not isinstance(weight, numbers.Real):
            weight = cast(numbers.Real, SCHEDULERS.build(weight))
        if bound is not None and bound <= 1e-4:
            raise ValueError('bound must be greater than 1e-4')
        super().__init__(**kwargs)
        self._reduction = reduction

        self._weight = weight
        self._threshold = None if bound is None else bound / weight

        self.register_forward_hook(
            self._forward_hook,  # type: ignore[arg-type]
        )

    @property
    def reduction(self) -> Reduction:
        return self._reduction

    def reduce(
        self,
        loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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

    def _forward_hook(
        self,
        module: nn.Module,
        input_: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self._threshold is None:
            return self._weight * output

        weight = self._weight
        # coef = bound / (weight * output)
        coef = self._threshold / output.item()
        # if bound < weight * output
        if coef < 1.0:
            # weight = bound / output
            weight *= coef

        output = weight * output
        return output


LOSSES: Registry[BaseLoss] = Registry(
    'losses',
    parent=STEPS,
    base=BaseLoss,
)
