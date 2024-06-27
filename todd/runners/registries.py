__all__ = [
    'CallbackRegistry',
    'StrategyRegistry',
    'ETARegistry',
]

from typing import Any

from ..bases.configs import Config
from ..bases.registries import Item, RegistryMeta
from ..registries import RunnerRegistry


class CallbackRegistry(RunnerRegistry):

    @classmethod
    def _build(cls, item: Item, config: Config) -> Any:
        from .callbacks import ComposedCallback
        if item is ComposedCallback:
            callbacks = config.callbacks
            runner = config.get('runner')
            config.priorities = [c.pop('priority', dict()) for c in callbacks]
            config.callbacks = [cls.build(c, runner=runner) for c in callbacks]
        return RegistryMeta._build(cls, item, config)


class StrategyRegistry(RunnerRegistry):
    pass


class ETARegistry(RunnerRegistry):
    pass
