__all__ = [
    'BaseCallback',
]

from typing import Any

from .. import BaseRunner, EpochBasedTrainer

Memo = dict[str, Any]


class BaseCallback:

    def should_break(self, runner: BaseRunner, batch, memo: Memo) -> bool:
        """Determine whether to break the run loop.

        Args:
            runner: the runner.
            batch: inputs.
            memo: runtime memory.

        Returns:
            Whether to break the run loop.

        Override this method for early stopping, error detection, etc.
        By default, this method returns `False` and the run loop ends normally
        when the dataloader is exhausted.
        """
        return False

    def should_continue(self, runner: BaseRunner, batch, memo: Memo) -> bool:
        """Determine whether to skip the current iteration.

        Args:
            runner: the runner.
            batch: inputs.
            memo: runtime memory.

        Returns:
            Whether to skip the current iteration.
        """
        return False

    def before_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        pass

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        pass

    def should_break_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> bool:
        return False

    def should_continue_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> bool:
        return False

    def before_run_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        pass

    def after_run_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        pass

    def before_run(self, runner: BaseRunner, memo: Memo) -> None:
        pass

    def after_run(self, runner: BaseRunner, memo: Memo) -> None:
        pass
