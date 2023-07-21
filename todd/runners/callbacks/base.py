__all__ = [
    'BaseCallback',
]

import contextlib
from typing import Any

from ...base import StateDictMixin
from ..runners import RunnerHolderMixin

Memo = dict[str, Any]


class BaseCallback(RunnerHolderMixin, StateDictMixin):

    def connect(self) -> None:
        pass

    def should_break(self, batch, memo: Memo) -> bool:
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

    def should_continue(self, batch, memo: Memo) -> bool:
        """Determine whether to skip the current iteration.

        Args:
            batch: inputs.
            memo: runtime memory.

        Returns:
            Whether to skip the current iteration.
        """
        return False

    def before_run_iter(self, batch, memo: Memo) -> None:
        pass

    def run_iter_context(
        self,
        exit_stack: contextlib.ExitStack,
        batch,
        memo: Memo,
    ) -> None:
        pass

    def after_run_iter(self, batch, memo: Memo) -> None:
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
