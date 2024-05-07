__all__ = [
    'OptimizerRegistry',
]

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from ..utils import Config, descendant_classes
from .registry import Item, Registry, RegistryMeta
from .reproducers import ReproducerRegistry

if TYPE_CHECKING:
    from ..reproducers.filters import NamedParametersFilter


class OptimizerRegistry(Registry):

    @staticmethod
    def params(model: nn.Module, config: Config) -> Config:
        config = config.copy()
        params = config.pop('params')
        filter_: 'NamedParametersFilter' = (
            ReproducerRegistry.child('FilterRegistry').build(params)
        )
        filtered_params = [p for _, p in filter_(model)]
        assert all(p.requires_grad for p in filtered_params)
        config.params = filtered_params
        return config

    @classmethod
    def _build(cls, item: Item, config: Config):
        model: nn.Module = config.pop('model')
        params = config.pop('params', None)
        if params is None:
            config.params = [p for p in model.parameters() if p.requires_grad]
        else:
            config.params = [cls.params(model, p) for p in params]
        return RegistryMeta._build(cls, item, config)


for c in descendant_classes(torch.optim.Optimizer):
    if '<locals>' not in c.__qualname__:
        OptimizerRegistry.register_()(cast(Item, c))
