__all__ = [
    'CheckpointCallback',
]

from typing import Any

import torch

from ...base import CallbackRegistry, Config
from ...utils import get_rank
from ..runners import BaseRunner, EpochBasedTrainer, Trainer
from .base import BaseCallback
from .interval import IntervalMixin

Memo = dict[str, Any]


@CallbackRegistry.register()
class CheckpointCallback(IntervalMixin, BaseCallback):

    def __init__(
        self,
        *args,
        state_dict: Config = Config(),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._state_dict = state_dict

    def _save(self, runner: Trainer, name: str) -> None:
        # for FSDP, all ranks should call state dict
        state_dict = runner.state_dict(**self._state_dict)

        if get_rank() != 0:
            return
        f = runner.work_dir / f'{name}.pth'
        runner.logger.info(f"Saving state dict to {f}")
        torch.save(state_dict, f)

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        if self._should_run_iter(runner):
            self._save(runner, f'iter_{runner.iter_}')

    def after_run_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        if self._should_run_epoch(runner):
            self._save(runner, f'epoch_{runner.epoch}')

    def after_run(self, runner: BaseRunner, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        self._save(runner, 'latest')
