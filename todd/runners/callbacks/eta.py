__all__ = [
    'ETACallback',
]

import datetime
from typing import Any

from ...base import CallbackRegistry
from .base import BaseCallback

Memo = dict[str, Any]


@CallbackRegistry.register()
class ETACallback(BaseCallback):

    def before_run(self, memo: Memo) -> None:
        super().before_run(memo)
        self._start_time = datetime.datetime.now()
        self._start_iter = self._runner.iter_

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        if 'log' in memo:
            eta = datetime.datetime.now() - self._start_time
            eta *= self._runner.iters - self._runner.iter_
            eta /= self._runner.iter_ - self._start_iter
            memo['log']['eta'] = str(eta)
