__all__ = [
    'LossMetric',
]

from typing import Any, Mapping, TypeVar

import einops
import torch
from torch import nn

from ...bases.configs import Config
from ...bases.registries import BuildPreHookMixin, Item, RegistryMeta
from ...models import LossRegistry
from ...models.losses import BaseLoss
from ...patches.py_ import get_
from ..memo import Memo
from ..registries import MetricRegistry
from .vanilla import Metric

T = TypeVar('T', bound=nn.Module)


@MetricRegistry.register_()
class LossMetric(BuildPreHookMixin, Metric[T]):

    def __init__(
        self,
        *args,
        loss: BaseLoss,
        inputs: Mapping[str, str],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._loss = loss
        self._inputs = dict(inputs)

    @classmethod
    def loss_build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        loss: BaseLoss = LossRegistry.build_or_return(
            config.loss,
            reduction='none',
        )
        loss.requires_grad_(False)
        loss.eval()
        config.loss = loss
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.loss_build_pre_hook(config, registry, item)
        return config

    def _forward(self, batch: Any, memo: Memo) -> tuple[torch.Tensor, Memo]:
        inputs = {k: get_(memo, v) for k, v in self._inputs.items()}
        loss = self._loss(**inputs)
        loss = einops.reduce(loss, 'b ... -> b', reduction='mean')
        return loss, memo
