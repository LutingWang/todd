__all__ = [
    'DatasetRegistry',
    'ClipGradRegistry',
    'CollateRegistry',
    'WorkerInitRegistry',
    'InitRegistry',
    'LRSchedulerRegistry',
    'OptimizerRegistry',
    'SamplerRegistry',
    'TransformRegistry',
    'DataLoaderRegistry',
]

from typing import TYPE_CHECKING, Any, cast

import torch
import torchvision.transforms as tf
from torch import nn
from torch.nn import init, utils
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import dataset

from ..bases.configs import Config
from ..bases.registries import Item, Registry, RegistryMeta
from ..loggers import logger
from ..patches.py import descendant_classes, get_classes
from .partial import PartialRegistry

if TYPE_CHECKING:
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
    def _build(cls, item: Item, config: Config) -> Any:
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
    def params(model: nn.Module, config: Config) -> Config:
        from ..models import FilterRegistry
        config = config.copy()
        params = config.pop('params')
        filter_: 'NamedParametersFilter' = FilterRegistry.build(params)
        filtered_params = [p for _, p in filter_(model)]
        assert all(p.requires_grad for p in filtered_params)
        config.params = filtered_params
        return config

    @classmethod
    def _build(cls, item: Item, config: Config) -> Any:
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


for c in get_classes(dataset, data.Dataset):
    DatasetRegistry.register_()(cast(Item, c))


def datasets_build_pre_hook(
    config: Config,
    registry: RegistryMeta,
    item: Item,
) -> Config:
    config.datasets = [
        DatasetRegistry.build_or_return(d) for d in config.datasets
    ]
    return config


DatasetRegistry.register_(
    force=True,
    build_pre_hook=datasets_build_pre_hook,
)(data.ConcatDataset)


class SamplerRegistry(Registry):
    pass


for c in descendant_classes(data.Sampler):
    SamplerRegistry.register_()(cast(Item, c))


class CollateRegistry(PartialRegistry):
    pass


class WorkerInitRegistry(PartialRegistry):
    pass


@WorkerInitRegistry.register_('default')
def default_worker_init(worker_id: int) -> None:
    from ..utils import init_seed
    logger.debug("Initializing worker %d", worker_id)
    init_seed(torch.initial_seed())


class TransformRegistry(Registry):
    pass


for c in get_classes(tf):
    TransformRegistry.register_()(cast(Item, c))


def transformers_build_pre_hook(
    config: Config,
    registry: RegistryMeta,
    item: Item,
) -> Config:
    config.transforms = [
        TransformRegistry.build_or_return(t) for t in config.transforms
    ]
    return config


TransformRegistry.register_(
    force=True,
    build_pre_hook=transformers_build_pre_hook,
)(tf.Compose)


class DataLoaderRegistry(Registry):

    @classmethod
    def _build(cls, item: Item, config: Config) -> Any:
        sampler = config.pop('sampler', None)
        if sampler is not None:
            config.sampler = SamplerRegistry.build(
                sampler,
                dataset=config.dataset,
            )

        batch_sampler = config.pop('batch_sampler', None)
        if batch_sampler is not None:
            config.batch_sampler = SamplerRegistry.build(
                batch_sampler,
                sampler=config.pop('sampler'),
            )

        collate_fn = config.pop('collate_fn', None)
        if collate_fn is not None:
            config.collate_fn = CollateRegistry.build(collate_fn)

        worker_init_fn = config.pop('worker_init_fn', None)
        if worker_init_fn is None:
            logger.info(
                "`worker_init_fn` is recommended to be `%s`, instead of `%s`.",
                'default',
                None,
            )
        else:
            config.worker_init_fn = WorkerInitRegistry.build(worker_init_fn)

        return RegistryMeta._build(cls, item, config)


DataLoaderRegistry.register_()(data.DataLoader)
