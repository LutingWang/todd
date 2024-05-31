__all__ = [
    'BaseStrategy',
]

from typing import Any, Mapping, TypeVar, cast

import torch
from torch import nn

from ...configs import Config
from ...patches.torch import get_rank
from ...registries import ModelRegistry, OptimizerRegistry
from ...utils import StateDictMixin
from ..registries import StrategyRegistry
from ..utils import RunnerHolderMixin

T = TypeVar('T', bound=nn.Module)


@StrategyRegistry.register_()
class BaseStrategy(RunnerHolderMixin[T], StateDictMixin):

    def __init__(
        self,
        *args,
        setup: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if setup is None:
            setup = Config()
        self.setup(setup)

    def setup(self, config: Config) -> None:
        pass

    def build_model(self, config: Config) -> nn.Module:
        return ModelRegistry.build(config)

    def map_model(self, model: nn.Module, config: Config) -> nn.Module:
        return model

    def wrap_model(self, model: nn.Module, config: Config) -> T:
        return cast(T, model)

    def build_optimizer(self, config: Config) -> torch.optim.Optimizer:
        return OptimizerRegistry.build(config, model=self.module)

    @property
    def module(self) -> nn.Module:
        return self.runner.model

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
            self.runner.logger.info(incompatible_keys)

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
                self.runner.logger.info("Loading model from %s", f_)
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
