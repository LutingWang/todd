__all__ = [
    'BaseStrategy',
]

from typing import Any, Generic, Mapping, TypeVar, cast

import torch
from torch import nn

from ...base import Config, ModelRegistry, OptimizerRegistry, StrategyRegistry
from ...utils import StateDictMixin, get_rank
from ..utils import RunnerHolderMixin

T = TypeVar('T', bound=nn.Module)


@StrategyRegistry.register_()
class BaseStrategy(RunnerHolderMixin, StateDictMixin, Generic[T]):

    def __init__(
        self,
        *args,
        model: Config,
        map_model: Config | None = None,
        wrap_model: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._build_model(model, map_model, wrap_model)

    def _build_model(
        self,
        config: Config,
        map_config: Config | None,
        wrap_config: Config | None,
    ) -> None:
        model = ModelRegistry.build(config)
        model = self.map_model(model, map_config)
        model = self.wrap_model(model, wrap_config)
        self._model = model

    def map_model(
        self,
        model: nn.Module,
        config: Config | None = None,
    ) -> nn.Module:
        if config is None:
            config = Config()
        return model

    def wrap_model(self, model: nn.Module, config: Config | None = None) -> T:
        if config is None:
            config = Config()
        return cast(T, model)

    def build_optimizer(self, config: Config) -> torch.optim.Optimizer:
        return OptimizerRegistry.build(config, model=self.module)

    @property
    def model(self) -> T:
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
