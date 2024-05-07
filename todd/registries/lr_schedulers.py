__all__ = [
    'LRSchedulerRegistry',
]

from typing import cast

from torch.optim import lr_scheduler

from ..configs import Config
from ..utils import descendant_classes
from .registry import Item, Registry, RegistryMeta


class LRSchedulerRegistry(Registry):

    @classmethod
    def _build(cls, item: Item, config: Config):
        if item is lr_scheduler.SequentialLR:
            config.schedulers = [
                cls.build(scheduler, optimizer=config.optimizer)
                for scheduler in config.schedulers
            ]
        return RegistryMeta._build(cls, item, config)


for c in descendant_classes(lr_scheduler.LRScheduler):
    LRSchedulerRegistry.register_()(cast(Item, c))
