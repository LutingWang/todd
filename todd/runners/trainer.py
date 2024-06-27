__all__ = [
    'Trainer',
]

from abc import ABC
from typing import TYPE_CHECKING, Any, Mapping, TypeVar

import torch
from torch import nn

from ..bases.configs import Config
from ..bases.registries import Builder, BuildSpec
from ..patches.py import classproperty
from ..registries import RunnerRegistry
from .base import BaseRunner
from .memo import Memo

if TYPE_CHECKING:
    from .strategies import BaseStrategy

T = TypeVar('T', bound=nn.Module)


@RunnerRegistry.register_()
class Trainer(BaseRunner[T], ABC):

    def __init__(
        self,
        *args,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> None:
        self._optimizer = optimizer
        super().__init__(*args, **kwargs)

    @classproperty
    def build_spec(self) -> BuildSpec:

        def build_optimizer(
            config: Config,
            strategy: 'BaseStrategy[T]',
            model: nn.Module,
        ) -> torch.optim.Optimizer:
            return strategy.build_optimizer(config, model)

        build_spec = BuildSpec(
            optimizer=Builder(
                build_optimizer,
                requires=dict(strategy='strategy', model='model'),
            ),
        )

        return super().build_spec | build_spec

    @property
    def iters_per_epoch(self) -> int:
        return len(self._dataloader)

    @property
    def inner_iter(self) -> int:
        return self._iter % self.iters_per_epoch

    @property
    def epoch(self) -> int:
        return self._iter // self.iters_per_epoch

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def _setup(self) -> Memo:
        self._model.train()
        return super()._setup()

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['optim'] = self._strategy.optim_state_dict(*args, **kwargs)
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        self._strategy.load_optim_state_dict(
            state_dict['optim'],
            *args,
            **kwargs,
        )
