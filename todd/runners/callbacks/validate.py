__all__ = [
    'ValidateCallback',
]

from typing import Any

import torch.distributed

from ...base import CallbackRegistry, Config, RunnerRegistry
from .. import BaseRunner, EpochBasedTrainer, Trainer, Validator
from .base import BaseCallback

Memo = dict[str, Any]


@CallbackRegistry.register()
class ValidateCallback(BaseCallback):

    def __init__(
        self,
        validator: Config,
        interval: int = 0,
        by_epoch: bool = False,
    ) -> None:
        self._validator: Validator = RunnerRegistry.build(validator)
        self._interval = interval
        self._by_epoch = by_epoch

    def _validate(self) -> None:
        torch.distributed.barrier()
        self._validator.run()

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        if self._by_epoch:
            return
        if self._interval > 0 and runner.iter_ % self._interval == 0:
            self._validate()

    def after_run_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        if not self._by_epoch:
            return
        if self._interval > 0 and runner.epoch % self._interval == 0:
            self._validate()

    def after_run(self, runner: BaseRunner, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        self._validate()
