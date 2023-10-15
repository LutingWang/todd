__all__ = [
    'FSDPStrategy',
]

from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...base import Config, OptimizerRegistry, StrategyRegistry
from .ddp import DDPStrategy

# TODO: update when pytorch updates


@StrategyRegistry.register_()
class FSDPStrategy(DDPStrategy):
    _model: FSDP  # type: ignore[assignment]

    def _wrap_model(self, config: Config) -> None:
        self._model = FSDP(self._model, **config)

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

    if TYPE_CHECKING:

        @property
        def model(self) -> FSDP:  # type: ignore[override]
            ...
