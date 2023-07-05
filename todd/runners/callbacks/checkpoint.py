__all__ = ['CheckpointCallback']

from typing import Any

import torch

from ...base import CallbackRegistry, Config
from .. import BaseRunner, EpochBasedTrainer, Trainer
from .base import BaseCallback

Memo = dict[str, Any]


@CallbackRegistry.register()
class CheckpointCallback(BaseCallback):

    def __init__(
        self,
        every_n_iters: int = 0,
        every_n_epochs: int = 0,
        state_dict: Config = Config(),
    ) -> None:
        self._every_n_iters = every_n_iters
        self._every_n_epochs = every_n_epochs
        self._state_dict = state_dict

    def _save(self, runner: Trainer, name: str) -> None:
        f = runner.work_dir / f'{name}.pth'
        runner.logger.info(f"Saving state dict to {f}")
        torch.save(runner.state_dict(**self._state_dict), f)

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        iter_ = runner.iter_
        if self._every_n_iters > 0 and iter_ % self._every_n_iters == 0:
            self._save(runner, f'iter_{iter_}')

    def after_run_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        epoch = runner.epoch
        if self._every_n_epochs > 0 and epoch % self._every_n_epochs == 0:
            self._save(runner, f'epoch_{epoch}')

    def after_run(self, runner: BaseRunner, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        self._save(runner, 'latest')
