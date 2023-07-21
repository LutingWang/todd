__all__ = [
    'LogCallback',
]

import logging
from typing import Any, cast

from ...base import CallbackRegistry, Formatter
from ...utils import get_rank, get_timestamp
from ..runners import EpochBasedTrainer
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

    def connect(self) -> None:
        super().connect()
        if get_rank() == 0 and self._with_file_handler:
            file = self._runner.work_dir / f'{get_timestamp()}.log'
            handler = logging.FileHandler(file)
            handler.setFormatter(Formatter())
            self._runner.logger.addHandler(handler)

    def before_run_iter(self, batch, memo: Memo) -> None:
        super().before_run_iter(batch, memo)
        if get_rank() == 0 and self._should_run_iter():
            memo['log'] = dict()

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        if 'log' not in memo:
            return
        log: dict[str, Any] = memo.pop('log')
        prefix = f"Iter [{self._runner.iter_}/{self._runner.iters}] "
        message = ' '.join(f'{k}={v}' for k, v in log.items())
        self._runner.logger.info(prefix + message)

    def before_run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super().before_run_epoch(epoch_memo, memo)
        runner = cast(EpochBasedTrainer, self._runner)
        if get_rank() == 0:
            runner.logger.info(f"Epoch [{runner.epoch}/{runner.epochs}]")
