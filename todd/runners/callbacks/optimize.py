__all__ = ['OptimizeCallback']

from typing import Any

import torch

from ...base import CallbackRegistry
from .. import BaseRunner, Trainer
from .base import BaseCallback

Memo = dict[str, Any]


@CallbackRegistry.register()
class OptimizeCallback(BaseCallback):

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        loss: torch.Tensor = memo['loss']
        loss.backward()
        runner.optimizer.step()
        runner.optimizer.zero_grad()
