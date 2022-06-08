import numbers
from typing import Literal, Optional, Union

import torch
from mmcv import ConfigDict
from mmcv.runner import BaseModule

from ..schedulers import SCHEDULERS

Reduction = Literal['none', 'mean', 'sum', 'prod']


class BaseLoss(BaseModule):

    def __init__(
        self,
        reduction: Reduction = 'mean',
        weight: Union[numbers.Real, ConfigDict] = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._reduction = reduction
        if not isinstance(weight, numbers.Real):
            weight = SCHEDULERS.build(weight)
        self._weight: numbers.Real = weight

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

    def weight(self, loss: torch.Tensor) -> torch.Tensor:
        return self._weight * loss
