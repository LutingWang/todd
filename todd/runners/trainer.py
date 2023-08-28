__all__ = [
    'Trainer',
]

from typing import Any, Mapping

import torch
import torch.distributed
import torch.utils.data

from ..base import Config, RunnerRegistry
from .base import BaseRunner

Memo = dict[str, Any]


@RunnerRegistry.register()
class Trainer(BaseRunner):

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
        self._strategy.model.train()
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
