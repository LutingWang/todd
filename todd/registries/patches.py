__all__ = [
    'DatasetRegistry',
    'ClipGradRegistry',
    'CollateRegistry',
    'InitRegistry',
    'LRSchedulerRegistry',
    'OptimizerRegistry',
    'SamplerRegistry',
    'TransformRegistry',
]

import inspect
from typing import TYPE_CHECKING, Any, cast

import torch
import torchvision.transforms as tf
from torch import nn
from torch.nn import init, utils
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import dataset

from ..patches.py import descendant_classes
from .partial import PartialRegistry
from .registry import BuildSpec, Item, Registry, RegistryMeta

if TYPE_CHECKING:
    from ..configs import Config
    from ..models.filters import NamedParametersFilter


class InitRegistry(PartialRegistry):
    pass


InitRegistry.register_()(init.uniform_)
InitRegistry.register_()(init.normal_)
InitRegistry.register_()(init.trunc_normal_)
InitRegistry.register_()(init.constant_)
InitRegistry.register_()(init.ones_)
InitRegistry.register_()(init.zeros_)
InitRegistry.register_()(init.eye_)
InitRegistry.register_()(init.dirac_)
InitRegistry.register_()(init.xavier_uniform_)
InitRegistry.register_()(init.xavier_normal_)
InitRegistry.register_()(init.kaiming_uniform_)
InitRegistry.register_()(init.kaiming_normal_)
InitRegistry.register_()(init.orthogonal_)
InitRegistry.register_()(init.sparse_)


class ClipGradRegistry(PartialRegistry):
    pass


ClipGradRegistry.register_()(utils.clip_grad_norm_)
ClipGradRegistry.register_()(utils.clip_grad_value_)


class LRSchedulerRegistry(Registry):

    @classmethod
    def _build(cls, item: Item, config: 'Config') -> Any:
        if item is lr_scheduler.SequentialLR:
            config.schedulers = [
                cls.build(scheduler, optimizer=config.optimizer)
                for scheduler in config.schedulers
            ]
        return RegistryMeta._build(cls, item, config)


for c in descendant_classes(lr_scheduler.LRScheduler):
    LRSchedulerRegistry.register_()(cast(Item, c))


class OptimizerRegistry(Registry):

    @staticmethod
    def params(model: nn.Module, config: 'Config') -> 'Config':
        from ..models import FilterRegistry
        config = config.copy()
        params = config.pop('params')
        filter_: 'NamedParametersFilter' = FilterRegistry.build(params)
        filtered_params = [p for _, p in filter_(model)]
        assert all(p.requires_grad for p in filtered_params)
        config.params = filtered_params
        return config

    @classmethod
    def _build(cls, item: Item, config: 'Config') -> Any:
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


class DatasetRegistry(Registry):
    pass


for _, c in inspect.getmembers(dataset, inspect.isclass):
    if issubclass(c, data.Dataset):
        DatasetRegistry.register_()(cast(Item, c))

DatasetRegistry.register_(
    force=True,
    build_spec=BuildSpec({'*datasets': DatasetRegistry.build}),
)(data.ConcatDataset)


class SamplerRegistry(Registry):
    pass


for c in descendant_classes(data.Sampler):
    SamplerRegistry.register_()(cast(Item, c))


class CollateRegistry(PartialRegistry):
    pass


class TransformRegistry(Registry):
    pass


for _, c in inspect.getmembers(tf, inspect.isclass):
    TransformRegistry.register_()(cast(Item, c))

TransformRegistry.register_(
    force=True,
    build_spec=BuildSpec({'*transforms': TransformRegistry.build}),
)(tf.Compose)
