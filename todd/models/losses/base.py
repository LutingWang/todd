__all__ = [
    'BaseLoss',
]

from abc import ABC, abstractmethod
from enum import StrEnum

import torch
from torch import nn

from ...bases.configs import Config
from ...bases.registries import BuildPreHookMixin, Item, RegistryMeta
from .schedulers import BaseScheduler, SchedulerRegistry


class Reduction(StrEnum):
    NONE = 'none'
    MEAN = 'mean'
    SUM = 'sum'
    PROD = 'prod'
    WEIGHTED = 'weighted'


class BaseLoss(BuildPreHookMixin, nn.Module, ABC):

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

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        if 'weight' in config:
            config.weight = SchedulerRegistry.build_or_return(config.weight)
        return config

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

        if self._reduction is Reduction.NONE:
            return loss
        if self._reduction is Reduction.WEIGHTED:
            assert mask is not None
            weight = mask.sum()
            if weight.abs() < 1e-6:
                return loss.new_zeros([])
            return loss.sum() / weight
        return getattr(loss, self._reduction.value)()

    def _scale(self, loss: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self._bound is not None:
            coef = self._bound / (weight * loss.item())
            weight *= min(coef, 1.)  # weight <= bound / loss
        return weight * loss

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass
