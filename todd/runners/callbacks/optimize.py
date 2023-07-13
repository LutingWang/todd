__all__ = [
    'OptimizeCallback',
]

from typing import Any, Mapping

import torch

from ...base import CallbackRegistry, Config
from ..runners import BaseRunner, Trainer
from .base import BaseCallback

Memo = dict[str, Any]


@CallbackRegistry.register()
class OptimizeCallback(BaseCallback):

    # TODO: add grad clip
    def __init__(
        self,
        *args,
        grad_scaler: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if grad_scaler is not None:
            self._build_grad_scaler(grad_scaler)

    @property
    def with_grad_scaler(self) -> bool:
        return hasattr(self, '_grad_scaler')

    def _build_grad_scaler(self, config: Config) -> None:
        self._grad_scaler = torch.cuda.amp.GradScaler(**config)

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        super().after_run_iter(runner, batch, memo)
        loss: torch.Tensor = memo['loss']
        if self.with_grad_scaler:
            loss = self._grad_scaler.scale(loss)
        loss.backward()
        if self.with_grad_scaler:
            self._grad_scaler.step(runner.optimizer)
            self._grad_scaler.update()
        else:
            runner.optimizer.step()
        runner.optimizer.zero_grad()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        if self.with_grad_scaler:
            self._grad_scaler.load_state_dict(state_dict['grad_scaler'])

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        if self.with_grad_scaler:
            state_dict['grad_scaler'] = self._grad_scaler.state_dict()
        return state_dict
