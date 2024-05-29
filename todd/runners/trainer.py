__all__ = [
    'Trainer',
]

from abc import ABC
from typing import Any, Mapping, TypeVar

import torch
from torch import nn

from ..configs import Config
from ..registries import RunnerRegistry
from .base import BaseRunner
from .memo import Memo

T = TypeVar('T', bound=nn.Module)


@RunnerRegistry.register_()
class Trainer(BaseRunner[T], ABC):

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

    def _build_optimizer(
        self,
        *args,
        optimizer: Config,
        **kwargs,
    ) -> None:
        self._optimizer = self._strategy.build_optimizer(optimizer)

    def _build(self, *args, **kwargs) -> None:
        super()._build(*args, **kwargs)
        self._build_optimizer(*args, **kwargs)

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
