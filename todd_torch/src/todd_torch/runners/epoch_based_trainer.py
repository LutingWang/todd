__all__ = [
    'EpochBasedTrainer',
]

import contextlib
import itertools
from collections import defaultdict
from typing import TypeVar

from torch import nn

from ..patches.torch import set_epoch
from ..registries import RunnerRegistry
from .memo import Memo
from .trainer import Trainer

T = TypeVar('T', bound=nn.Module)


@RunnerRegistry.register_()
class EpochBasedTrainer(Trainer[T]):

    def __init__(self, *args, epochs: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._epochs = epochs

    @property
    def iters(self) -> int:
        return self.iters_per_epoch * self._epochs

    @property
    def epochs(self) -> int:
        return self._epochs

    def _run_epoch(self, epoch_memo: Memo, memo: Memo) -> Memo:
        return super()._run(epoch_memo)

    def _setup_epoch(self, memo: Memo) -> Memo:
        epoch_memo = super()._setup()
        set_epoch(self._dataloader, self.epoch)
        epoch_memo.update(
            dataloader=(
                itertools.islice(self._dataloader, self.inner_iter, None)
                if self.inner_iter > 0 else self._dataloader
            ),
            epoch=defaultdict(list),
        )
        return epoch_memo

    def _teardown_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super()._teardown(epoch_memo)
        memo['epoch_memos'][self.epoch] = epoch_memo['epoch']

    def _run(self, memo: Memo) -> Memo:
        while self.epoch < self._epochs:
            epoch_memo = self._setup_epoch(memo)

            if self._callbacks.should_break_epoch(epoch_memo, memo):
                break
            if self._callbacks.should_continue_epoch(epoch_memo, memo):
                continue

            self._callbacks.before_run_epoch(epoch_memo, memo)
            with contextlib.ExitStack() as exit_stack:
                self._callbacks.run_epoch_context(
                    exit_stack,
                    epoch_memo,
                    memo,
                )
                epoch_memo = self._run_epoch(epoch_memo, memo)
            self._callbacks.after_run_epoch(epoch_memo, memo)

            self._teardown_epoch(epoch_memo, memo)
        return memo

    def _setup(self) -> Memo:
        return dict(epoch_memos=dict())

    def _teardown(self, memo: Memo) -> None:
        pass
