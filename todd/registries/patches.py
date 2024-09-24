__all__ = [
    'InitRegistry',
    'ClipGradRegistry',
    'LRSchedulerRegistry',
    'OptimizerRegistry',
    'TransformRegistry',
    'DatasetRegistry',
    'SamplerRegistry',
    'CollateRegistry',
    'WorkerInitRegistry',
    'DataLoaderRegistry',
]

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.utils.data.dataset
import torchvision.transforms.v2 as tf
from torch import nn
from torch.nn import init, utils
from torch.optim import lr_scheduler

from ..bases.configs import Config
from ..bases.registries import Item, Registry, RegistryMeta
from ..loggers import logger
from ..patches.py_ import descendant_classes, get_classes
from ..patches.torch import PrefetchDataLoader, get_world_size
from .partial import PartialRegistry

if TYPE_CHECKING:
    from ..models.filters import NamedParametersFilter


class InitRegistry(PartialRegistry):
    pass


register = InitRegistry.register_()
register(init.uniform_)
register(init.normal_)
register(init.trunc_normal_)
register(init.constant_)
register(init.ones_)
register(init.zeros_)
register(init.eye_)
register(init.dirac_)
register(init.xavier_uniform_)
register(init.xavier_normal_)
register(init.kaiming_uniform_)
register(init.kaiming_normal_)
register(init.orthogonal_)
register(init.sparse_)


class ClipGradRegistry(PartialRegistry):
    pass


register = ClipGradRegistry.register_()
register(utils.clip_grad_norm_)
register(utils.clip_grad_value_)


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


class DatasetRegistry(Registry):
    pass


for c in get_classes(torch.utils.data.dataset, torch.utils.data.Dataset):
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
)(torch.utils.data.ConcatDataset)


class SamplerRegistry(Registry):
    pass


for c in descendant_classes(torch.utils.data.Sampler):
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


class DataLoaderRegistry(Registry):
    pass


def dataloader_build_pre_hook(
    config: Config,
    registry: RegistryMeta,
    item: Item,
) -> Config:
    dataset = DatasetRegistry.build_or_return(config.dataset)
    config.dataset = dataset

    if config.pop('batch_size_in_total', False):
        batch_size = config.batch_size
        world_size = get_world_size()
        assert batch_size % world_size == 0
        config.batch_size = batch_size // world_size

    if (sampler := config.get('sampler')) is not None:
        config.sampler = SamplerRegistry.build(sampler, dataset=dataset)

    if (batch_sampler := config.get('batch_sampler')) is not None:
        sampler = config.pop('sampler')
        config.batch_sampler = SamplerRegistry.build_or_return(
            batch_sampler,
            sampler=sampler,
        )

    if (collate_fn := config.get('collate_fn')) is not None:
        config.collate_fn = CollateRegistry.build_or_return(collate_fn)

    if (worker_init_fn := config.get('worker_init_fn')) is not None:
        config.worker_init_fn = WorkerInitRegistry.build(worker_init_fn)
    else:
        logger.info(
            "`worker_init_fn` is recommended to be '%s', instead of %s.",
            'default',
            None,
        )

    return config


register = DataLoaderRegistry.register_(
    build_pre_hook=dataloader_build_pre_hook,
)
register(torch.utils.data.DataLoader)
register(PrefetchDataLoader)
