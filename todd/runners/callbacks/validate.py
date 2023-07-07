__all__ = [
    'ValidateCallback',
]

from typing import Any

import torch.distributed

from ...base import CallbackRegistry, Config, RunnerRegistry
from .. import BaseRunner, EpochBasedTrainer, Trainer, Validator
from .base import BaseCallback
from .interval import IntervalMixin

Memo = dict[str, Any]


@CallbackRegistry.register()
class ValidateCallback(IntervalMixin, BaseCallback):

    def __init__(
        self,
        *args,
        validator: Config,
        by_epoch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._validator: Validator = RunnerRegistry.build(validator)
        self._by_epoch = by_epoch

    def _validate(self) -> None:
        torch.distributed.barrier()
        self._validator.run()

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        if self._by_epoch:
            return
        if self._should_run_iter(runner):
            self._validate()

    def after_run_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        if not self._by_epoch:
            return
        if self._should_run_epoch(runner):
            self._validate()

    def after_run(self, runner: BaseRunner, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        self._validate()
