__all__ = [
    'EpochBasedTrainer',
]

import contextlib
from typing import Any, Mapping

import torch
import torch.distributed
import torch.utils.data

from .trainer import Trainer

Memo = dict[str, Any]


class EpochBasedTrainer(Trainer):

    def __init__(self, *args, epochs: int, **kwargs) -> None:
        self._epochs = epochs

        # must be set before _callbacks.connect() to allow loading state dict
        self._epoch = 0

        super().__init__(*args, **kwargs)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def iters(self) -> int:
        return super().iters * self._epochs

    @property
    def epochs(self) -> int:
        return self._epochs

    def _run_epoch(self, epoch_memo: Memo, memo: Memo) -> Memo:
        return super()._run(epoch_memo)

    def _setup_epoch(self, memo: Memo) -> Memo:
        sampler = self._dataloader.batch_sampler
        if sampler is None:
            sampler = self._dataloader.sampler
        if isinstance(sampler, torch.utils.data.DistributedSampler):
            sampler.set_epoch(self._epoch)
        return super()._setup()

    def _teardown_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super()._teardown(epoch_memo)
        memo['epoch_memos'][self._epoch] = epoch_memo

    def _run(self, memo: Memo) -> Memo:
        while self._epoch < self._epochs:
            self._epoch += 1
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

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        self._epoch = state_dict['meta']['epoch']

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['meta']['epoch'] = self._epoch
        return state_dict
