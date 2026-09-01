__all__ = [
    'MonitorCallback',
]

import contextlib
import logging
from typing import Any, TypeVar

from torch import nn

from ..memo import Memo
from ..registries import CallbackRegistry
from .base import BaseCallback

T = TypeVar('T', bound=nn.Module)


class Context:

    def __init__(self, logger: logging.Logger, **kwargs) -> None:
        self._logger = logger
        self._kwargs = kwargs

    def __enter__(self) -> None:
        pass

    def __exit__(self, *exc_info) -> None:
        if all(i is None for i in exc_info):
            return
        message = '\n'.join(f'{k}={v}' for k, v in self._kwargs.items())
        self._logger.exception("Unable to run " + message)


@CallbackRegistry.register_()
class MonitorCallback(BaseCallback[T]):

    def run_iter_context(
        self,
        exit_stack: contextlib.ExitStack,
        batch: Any,
        memo: Memo,
    ) -> None:
        super().run_iter_context(exit_stack, batch, memo)
        context = Context(
            self.runner.logger,
            iter_=self.runner.iter_,
            batch=batch,
            memo=memo,
        )
        exit_stack.enter_context(context)

    def run_epoch_context(
        self,
        exit_stack: contextlib.ExitStack,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        super().run_epoch_context(exit_stack, epoch_memo, memo)
        runner = self.epoch_based_trainer
        context = Context(
            runner.logger,
            epoch=runner.epoch,
            epoch_memo=epoch_memo,
            memo=memo,
        )
        exit_stack.enter_context(context)
