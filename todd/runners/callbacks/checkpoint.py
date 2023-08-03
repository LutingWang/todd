__all__ = [
    'CheckpointCallback',
]

import pathlib
from typing import Any

import torch

from ...base import CallbackRegistry, Config
from ...utils import get_rank
from .base import BaseCallback
from .interval import IntervalMixin

Memo = dict[str, Any]


@CallbackRegistry.register()
class CheckpointCallback(IntervalMixin, BaseCallback):

    def __init__(
        self,
        *args,
        state_dict: Config | None = None,
        load_state_dict: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trainer
        if state_dict is None:
            state_dict = Config()
        self._state_dict = state_dict
        if load_state_dict is None:
            load_state_dict = Config()
        self._load_state_dict = load_state_dict

    def init(self) -> None:
        super().init()
        self._checkpoint_dir = self._runner.work_dir / 'checkpoints'
        self._latest_checkpoint_dir = self._checkpoint_dir / 'latest'

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self._runner.load_from is not None:
            load_from = pathlib.Path(self._runner.load_from)
            assert load_from.exists()
            self._runner._logger.info(f"Loading from {load_from}")
            state_dict = {
                f.stem: torch.load(f, 'cpu')
                for f in load_from.glob('*.pth')
            }
            self._runner.load_state_dict(state_dict, **self._load_state_dict)

    def _save(self, name: str) -> None:
        # for FSDP, all ranks should call state dict
        state_dict = self._runner.state_dict(**self._state_dict)

        if get_rank() != 0:
            return
        work_dir = self._checkpoint_dir / name
        work_dir.mkdir(parents=True, exist_ok=True)
        self._runner.logger.info(f"Saving state dict to {work_dir}")
        for k, v in state_dict.items():
            torch.save(v, work_dir / f'{k}.pth')

        if self._latest_checkpoint_dir.is_symlink():
            self._latest_checkpoint_dir.unlink()
        self._latest_checkpoint_dir.symlink_to(work_dir)

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        if self._should_run_iter():
            self._save(f'iter_{self._runner.iter_}')

    def after_run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super().after_run_epoch(epoch_memo, memo)
        if self._should_run_epoch():
            self._save(f'epoch_{self.epoch_based_trainer.epoch}')
