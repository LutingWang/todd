__all__ = [
    'FSDPStrategy',
]

from typing import Any, Mapping, TypeVar, cast

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...configs import Config
from ...registries import OptimizerRegistry
from ..registries import StrategyRegistry
from .cuda import CUDAStrategy

# TODO: update when pytorch updates

T = TypeVar('T', bound=FSDP)


@StrategyRegistry.register_()
class FSDPStrategy(CUDAStrategy[T]):

    def wrap_model(self, model: nn.Module, config: Config) -> T:
        model = super().wrap_model(model, config)
        model = FSDP(model, **config)
        return cast(T, model)

    @property
    def module(self) -> nn.Module:
        return self.runner.model.module

    def build_optimizer(self, config: Config) -> torch.optim.Optimizer:
        return OptimizerRegistry.build(config, model=self.runner.model)

    def model_state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return self.runner.model.state_dict(*args, **kwargs)

    def load_model_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        self.runner.model.load_state_dict(state_dict, *args, **kwargs)

    def optim_state_dict(
        self,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        trainer = self.trainer
        return FSDP.full_optim_state_dict(trainer.model, trainer.optimizer)

    def load_optim_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        state_dict = dict(state_dict)
        trainer = self.trainer
        sharded_state_dict = FSDP.scatter_full_optim_state_dict(
            state_dict,
            trainer.model,
        )
        trainer.optimizer.load_state_dict(sharded_state_dict)
