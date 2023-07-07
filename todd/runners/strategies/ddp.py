__all__ = [
    'DDPStrategy',
]

import torch.nn as nn

from ...base import StrategyRegistry, get_local_rank
from .base import BaseStrategy


@StrategyRegistry.register()
class DDPStrategy(BaseStrategy):

    def wrap_model(
        self,
        model: nn.Module,
    ) -> nn.parallel.DistributedDataParallel:
        return nn.parallel.DistributedDataParallel(
            model,
            [get_local_rank()],
        )

    def get_model(
        self,
        model: nn.parallel.DistributedDataParallel,
    ) -> nn.Module:
        return model.module
