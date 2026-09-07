__all__ = [
    'Metric',
]

from abc import abstractmethod
from typing import Any, TypeVar

import torch
from torch import nn

from ...patches.torch import all_gather_object
from ..memo import Memo
from .base import BaseMetric

T = TypeVar('T', bound=nn.Module)


class Metric(BaseMetric[T]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._values: list[torch.Tensor] = []

    @abstractmethod
    def _forward(self, batch: Any, memo: Memo) -> tuple[torch.Tensor, Memo]:
        pass

    def forward(self, batch: Any, memo: Memo) -> Memo:
        log: Memo | None = memo.get('log')
        values, memo = self._forward(batch, memo)
        if values.shape == tuple():  # to support torch.cat
            values = values.view(1)
        self._values.append(values)
        if log is not None:
            log[self._name] = f'{values.mean():.3f}'
        return memo

    def summary(self, memo: Memo) -> float:
        values = torch.cat(self._values)
        values = torch.cat(all_gather_object(values))
        values = values.mean()
        return values.item()
