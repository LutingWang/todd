__all__ = ['OptimizeCallback']

from typing import Any

import torch

from ...base import CallbackRegistry
from .. import BaseRunner, Trainer
from .base import BaseCallback

Memo = dict[str, Any]


@CallbackRegistry.register()
class OptimizeCallback(BaseCallback):

    # TODO: add grad clip

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        loss: torch.Tensor = memo['loss']
        if runner.with_grad_scaler:
            loss = runner.grad_scaler.scale(loss)
        loss.backward()
        if runner.with_grad_scaler:
            runner.grad_scaler.step(runner.optimizer)
            runner.grad_scaler.update()
        else:
            runner.optimizer.step()
        runner.optimizer.zero_grad()
