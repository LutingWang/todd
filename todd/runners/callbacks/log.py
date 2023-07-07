__all__ = [
    'LogCallback',
]

import logging
from typing import Any

from ...base import CallbackRegistry, Formatter, get_rank
from ...utils import get_timestamp
from .. import BaseRunner, EpochBasedTrainer
from .base import BaseCallback
from .interval import IntervalMixin

Memo = dict[str, Any]


@CallbackRegistry.register()
class LogCallback(IntervalMixin, BaseCallback):

    def __init__(
        self,
        *args,
        with_file_handler: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._with_file_handler = with_file_handler

    def before_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        if get_rank() == 0 and self._should_run_iter(runner):
            memo['log'] = dict()

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        if 'log' not in memo:
            return
        log: dict[str, Any] = memo.pop('log')
        prefix = f"Iter [{runner.iter_}/{runner.iters}] "
        message = ' '.join(f'{k}={v}' for k, v in log.items())
        runner.logger.info(prefix + message)

    def before_run_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        runner.logger.info(f"Epoch [{runner.epoch}/{runner.epochs}]")

    def before_run(self, runner: BaseRunner, memo: Memo) -> None:
        if get_rank() == 0 and self._with_file_handler:
            file = runner.work_dir / f'{get_timestamp()}.log'
            handler = logging.FileHandler(file)
            handler.setFormatter(Formatter())
            runner.logger.addHandler(handler)
            self._handler = handler

    def after_run(self, runner: BaseRunner, memo: Memo) -> None:
        if get_rank() == 0 and self._with_file_handler:
            runner.logger.removeHandler(self._handler)
