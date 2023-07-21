__all__ = [
    'BaseStrategy',
]

import pathlib
from typing import Any, Mapping

import torch
import torch.nn as nn

from ...base import (
    Config,
    ModelRegistry,
    OptimizerRegistry,
    StateDictMixin,
    StrategyRegistry,
)
from ..runners import RunnerHolderMixin


@StrategyRegistry.register()
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
        return OptimizerRegistry.build(config, model=self._model)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def module(self) -> nn.Module:
        return self._model

    def model_state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return self._model.state_dict(*args, **kwargs)

    def load_model_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        self._model.load_state_dict(state_dict, *args, **kwargs)

    def load_model_from(self, f: pathlib.Path, *args, **kwargs) -> None:
        self._runner._logger.info(f"Loading model from {f}")
        model_state_dict = torch.load(f, 'cpu')
        self.load_model_state_dict(model_state_dict, *args, **kwargs)

    def optim_state_dict(
        self,
        optimizer: torch.optim.Optimizer,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        return optimizer.state_dict()

    def load_optim_state_dict(
        self,
        optimizer: torch.optim.Optimizer,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        optimizer.load_state_dict(state_dict)
