__all__ = [
    'BaseStrategy',
]

from typing import Any, Mapping

import torch
from torch import nn

from ...base import Config, ModelRegistry, OptimizerRegistry, StrategyRegistry
from ...utils import StateDictMixin, get_rank
from ..utils import RunnerHolderMixin


@StrategyRegistry.register_()
class BaseStrategy(RunnerHolderMixin, StateDictMixin):
    _model: nn.Module

    def __init__(
        self,
        *args,
        model: Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._build_model(model)

    def _build_model(self, config: Config) -> None:
        self._model = ModelRegistry.build(config)

    def build_optimizer(self, config: Config) -> torch.optim.Optimizer:
        return OptimizerRegistry.build(config, model=self.module)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def module(self) -> nn.Module:
        return self._model

    def model_state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return self.module.state_dict(*args, **kwargs)

    def load_model_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        incompatible_keys = self.module.load_state_dict(
            state_dict,
            *args,
            **kwargs,
        )
        if get_rank() == 0:
            self._runner.logger.info(incompatible_keys)

    def load_model_from(
        self,
        f: (
            torch.serialization.FILE_LIKE
            | list[torch.serialization.FILE_LIKE]
        ),
        *args,
        **kwargs,
    ) -> None:
        f_list = f if isinstance(f, list) else [f]
        model_state_dict = dict()
        for f_ in f_list:
            if get_rank() == 0:
                self._runner.logger.info("Loading model from %s", f_)
            model_state_dict.update(torch.load(f_, 'cpu'))
        self.load_model_state_dict(model_state_dict, *args, **kwargs)

    def optim_state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return self.trainer.optimizer.state_dict()

    def load_optim_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        state_dict = dict(state_dict)
        self.trainer.optimizer.load_state_dict(state_dict)
