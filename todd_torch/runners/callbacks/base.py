# pylint: disable=pointless-statement

__all__ = [
    'BaseCallback',
]

import contextlib
from typing import Any, TypeVar

from torch import nn

from ...utils import StateDictMixin
from ..memo import Memo
from ..registries import CallbackRegistry
from ..utils import RunnerHolderMixin

T = TypeVar('T', bound=nn.Module)


@CallbackRegistry.register_()
class BaseCallback(RunnerHolderMixin[T], StateDictMixin):

    def should_break(self, batch: Any, memo: Memo) -> bool:
        """Determine whether to break the run loop.

        Args:
            batch: inputs.
            memo: runtime memory.

        Returns:
            Whether to break the run loop.

        Override this method for early stopping, error detection, etc.
        By default, this method returns `False` and the run loop ends normally
        when the dataloader is exhausted.
        """
        return False

    def should_continue(self, batch: Any, memo: Memo) -> bool:
        """Determine whether to skip the current iteration.

        Args:
            batch: inputs.
            memo: runtime memory.

        Returns:
            Whether to skip the current iteration.
        """
        return False

    def before_run_iter(self, batch: Any, memo: Memo) -> None:
        pass

    def run_iter_context(
        self,
        exit_stack: contextlib.ExitStack,
        batch: Any,
        memo: Memo,
    ) -> None:
        pass

    def after_run_iter(self, batch: Any, memo: Memo) -> None:
        pass

    def should_break_epoch(self, epoch_memo: Memo, memo: Memo) -> bool:
        self.epoch_based_trainer
        return False

    def should_continue_epoch(self, epoch_memo: Memo, memo: Memo) -> bool:
        self.epoch_based_trainer
        return False

    def before_run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        self.epoch_based_trainer

    def run_epoch_context(
        self,
        exit_stack: contextlib.ExitStack,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        self.epoch_based_trainer

    def after_run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        self.epoch_based_trainer

    def before_run(self, memo: Memo) -> None:
        pass

    def after_run(self, memo: Memo) -> None:
        pass
