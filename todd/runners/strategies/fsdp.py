__all__ = [
    'FSDPStrategy',
]

from typing import Any, Mapping, TypeVar, cast

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...base import Config, OptimizerRegistry, StrategyRegistry
from .cuda import CUDAStrategy

# TODO: update when pytorch updates

T = TypeVar('T', bound=FSDP)


@StrategyRegistry.register_()
class FSDPStrategy(CUDAStrategy[T]):

    def wrap_model(self, model: nn.Module, config: Config | None = None) -> T:
        if config is None:
            config = Config()
        model = super().wrap_model(model, config)
        model = FSDP(model, **config)
        return cast(T, model)

    @property
    def module(self) -> nn.Module:
        return self._model.module

    def build_optimizer(self, config: Config) -> torch.optim.Optimizer:
        return OptimizerRegistry.build(config, model=self._model)

    def model_state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return self._model.state_dict(*args, **kwargs)

    def load_model_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        self._model.load_state_dict(state_dict, *args, **kwargs)

    def optim_state_dict(
        self,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        return FSDP.full_optim_state_dict(self._model, self.trainer.optimizer)

    def load_optim_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        state_dict = dict(state_dict)
        sharded_state_dict = FSDP.scatter_full_optim_state_dict(
            state_dict,
            self._model,
        )
        self.trainer.optimizer.load_state_dict(sharded_state_dict)
