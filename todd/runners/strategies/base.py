__all__ = [
    'BaseStrategy',
]

from typing import Any, Mapping

import torch.nn as nn

from ...base import Config, ModelRegistry, StateDict, StrategyRegistry
from ..runners import Trainer


@StrategyRegistry.register()
class BaseStrategy(StateDict):
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

    def optim_state_dict(
        self,
        runner: Trainer,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        return runner.optimizer.state_dict()

    def load_optim_state_dict(
        self,
        runner: Trainer,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        runner.optimizer.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['model'] = self.model_state_dict(*args, **kwargs)
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        self.load_model_state_dict(state_dict['model'], *args, **kwargs)
