__all__ = [
    'DDPStrategy',
]

import torch.nn as nn

from ...base import Config, StrategyRegistry
from .base import BaseStrategy


@StrategyRegistry.register()
class DDPStrategy(BaseStrategy):

    def __init__(self, *args, wrap: Config | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wrap = Config() if wrap is None else wrap

    def wrap_model(
        self,
        model: nn.Module,
    ) -> nn.parallel.DistributedDataParallel:
        return nn.parallel.DistributedDataParallel(model, **self._wrap)

    def get_model(
        self,
        model: nn.parallel.DistributedDataParallel,
    ) -> nn.Module:
        return model.module
