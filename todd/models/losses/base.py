__all__ = [
    'BaseLoss',
]

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
from torch import nn

from ...patches.py import classproperty
from ...registries import BuildSpec, BuildSpecMixin
from .schedulers import BaseScheduler, SchedulerRegistry


class Reduction(StrEnum):
    NONE = 'none'
    MEAN = 'mean'
    SUM = 'sum'
    PROD = 'prod'


class BaseLoss(BuildSpecMixin, nn.Module, ABC):

    def __init__(
        self,
        reduction: str | Reduction = Reduction.MEAN,
        weight: float | BaseScheduler = 1.0,
        bound: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._reduction = (
            reduction if isinstance(reduction, Reduction) else
            Reduction(reduction.lower())
        )
        self._weight = (
            weight if isinstance(weight, BaseScheduler) else
            BaseScheduler(gain=weight)
        )
        self._bound = bound
        self.register_forward_hook(lambda m, i, o: self._scale(o))

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(weight=SchedulerRegistry.build)
        return super().build_spec | build_spec

    @property
    def reduction(self) -> Reduction:
        return self._reduction

    @property
    def weight(self) -> float:
        return self._weight()

    def step(self) -> None:
        self._weight.step()

    def _reduce(
        self,
        loss: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            loss = loss * mask
        if self._reduction is not Reduction.NONE:
            loss = getattr(loss, self._reduction.value)()
        return loss

    def _scale(self, loss: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self._bound is not None:
            coef = self._bound / (weight * loss.item())
            weight *= min(coef, 1.)  # weight <= bound / loss
        return weight * loss

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass

    if TYPE_CHECKING:
        __call__ = forward
