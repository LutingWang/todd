__all__ = [
    'AccuracyMetric',
]

from typing import Any, TypeVar

import einops
import torch
from torch import nn

from ...patches.py_ import get_
from ..memo import Memo
from ..registries import MetricRegistry
from .vanilla import Metric

T = TypeVar('T', bound=nn.Module)


@MetricRegistry.register_()
class AccuracyMetric(Metric[T]):

    def __init__(
        self,
        *args,
        top_k: int,
        logits: str,
        target: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._top_k = top_k
        self._logits = logits
        self._target = target

    def _forward(self, batch: Any, memo: Memo) -> tuple[torch.Tensor, Memo]:
        logits: torch.Tensor = get_(memo, self._logits)
        target: torch.Tensor = get_(memo, self._target)
        assert logits.dtype.is_floating_point
        assert not target.dtype.is_floating_point
        _, pred = logits.topk(self._top_k)
        target = einops.rearrange(target, '... -> ... 1')
        accuracy: torch.Tensor = pred == target
        accuracy = accuracy.float().sum(-1)
        return accuracy, memo
