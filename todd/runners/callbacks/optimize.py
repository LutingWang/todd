__all__ = [
    'OptimizeCallback',
]

from typing import Any, Mapping

import torch

from ...base import CallbackRegistry, Config, GradClipperRegistry
from .base import BaseCallback

Memo = dict[str, Any]


@CallbackRegistry.register()
class OptimizeCallback(BaseCallback):

    # TODO: add accumulate
    def __init__(
        self,
        *args,
        grad_scaler: Config | None = None,
        grad_clip: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainer
        if grad_scaler is not None:
            self._build_grad_scaler(grad_scaler)
        if grad_clip is not None:
            self._build_grad_clipper(grad_clip)

    @property
    def with_grad_scaler(self) -> bool:
        return hasattr(self, '_grad_scaler')

    @property
    def with_grad_clipper(self) -> bool:
        return hasattr(self, '_grad_clipper')

    def _build_grad_scaler(self, config: Config) -> None:
        self._grad_scaler = torch.cuda.amp.GradScaler(**config)

    def _build_grad_clipper(self, config: Config) -> None:
        self._grad_clipper = GradClipperRegistry.build(config)

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        runner = self.trainer
        loss: torch.Tensor = memo['loss']
        if self.with_grad_scaler:
            loss = self._grad_scaler.scale(loss)
        loss.backward()
        if self.with_grad_clipper is not None:
            if self.with_grad_scaler:
                self._grad_scaler.unscale_(runner.optimizer)
            parameters = [
                param for param_group in runner.optimizer.param_groups
                for param in param_group['params']
            ]
            grad = self._grad_clipper(parameters)
            if 'log' in memo:
                memo['log']['grad'] = grad
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
