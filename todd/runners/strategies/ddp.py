__all__ = [
    'DDPStrategy',
]

from typing import TypeVar, cast

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ...bases.configs import Config
from ..registries import StrategyRegistry
from .cuda import CUDAStrategy

T = TypeVar('T', bound=DDP)


@StrategyRegistry.register_()
class DDPStrategy(CUDAStrategy[T]):

    def wrap_model(self, model: nn.Module, config: Config) -> T:
        model = super().wrap_model(model, config)
        model = DDP(model, **config)
        return cast(T, model)

    @property
    def module(self) -> nn.Module:
        return self.runner.model.module
