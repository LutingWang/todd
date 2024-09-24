__all__ = [
    'ReadyMadeMetric',
]

from typing import Any, TypeVar

import torch
from torch import nn

from ...patches.py_ import get_
from ..memo import Memo
from ..registries import MetricRegistry
from .vanilla import Metric

T = TypeVar('T', bound=nn.Module)


@MetricRegistry.register_()
class ReadyMadeMetric(Metric[T]):

    def __init__(self, *args, attr: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._attr = attr

    def _forward(self, batch: Any, memo: Memo) -> tuple[torch.Tensor, Memo]:
        metric = get_(memo, self._attr)
        return metric, memo
