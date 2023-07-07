__all__ = ["LrScheduleCallback"]

from typing import Any

from ...base import CallbackRegistry
from .. import BaseRunner, EpochBasedTrainer, Trainer
from .base import BaseCallback

Memo = dict[str, Any]


@CallbackRegistry.register()
class LrScheduleCallback(BaseCallback):

    def __init__(self, *args, by_epoch: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._by_epoch = by_epoch

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        if not self._by_epoch:
            runner.lr_scheduler.step()
        if 'log' in memo:
            memo['log']['lr'] = [
                f'{lr:.3e}' for lr in runner.lr_scheduler.get_last_lr()
            ]

    def after_run_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        if self._by_epoch:
            runner.lr_scheduler.step()
