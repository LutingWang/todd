__all__ = [
    'MonitorCallback',
]

import contextlib
from typing import Any

from ...base import CallbackRegistry
from ..runners import BaseRunner, EpochBasedTrainer
from .base import BaseCallback

Memo = dict[str, Any]


class Context:

    def __init__(self, runner: BaseRunner, **kwargs) -> None:
        self._logger = runner.logger
        self._kwargs = kwargs

    def __enter__(self) -> None:
        pass

    def __exit__(self, *exc_info) -> None:
        if all(i is None for i in exc_info):
            return
        message = '\n'.join(f'{k}={v}' for k, v in self._kwargs.items())
        self._logger.exception("Unable to run " + message)


@CallbackRegistry.register()
class MonitorCallback(BaseCallback):

    def run_iter_context(
        self,
        runner: BaseRunner,
        exit_stack: contextlib.ExitStack,
        batch,
        memo: Memo,
    ) -> None:
        super().run_iter_context(runner, exit_stack, batch, memo)
        context = Context(
            runner,
            iter_=runner.iter_,
            batch=batch,
            memo=memo,
        )
        exit_stack.enter_context(context)

    def run_epoch_context(
        self,
        runner: EpochBasedTrainer,
        exit_stack: contextlib.ExitStack,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        super().run_epoch_context(runner, exit_stack, epoch_memo, memo)
        context = Context(
            runner,
            epoch=runner.epoch,
            epoch_memo=epoch_memo,
            memo=memo,
        )
        exit_stack.enter_context(context)
