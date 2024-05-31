# pylint: disable=pointless-statement

__all__ = [
    'CheckpointCallback',
]

import pathlib

import torch

from ...configs import Config
from ...patches.torch import get_rank
from ..memo import Memo
from ..registries import CallbackRegistry
from .base import BaseCallback
from .interval import IntervalMixin


@CallbackRegistry.register_()
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

    def init(self, *args, **kwargs) -> None:
        super().init(*args, **kwargs)
        self._checkpoint_dir = self.runner.work_dir / 'checkpoints'
        self._latest_checkpoint_dir = self._checkpoint_dir / 'latest'

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.runner._auto_resume and self._latest_checkpoint_dir.exists():
            load_from = self._latest_checkpoint_dir
        elif self.runner.load_from is not None:
            load_from = pathlib.Path(self.runner.load_from)
            assert load_from.exists()
        else:
            load_from = None

        if load_from is not None:
            if get_rank() == 0:
                self.runner.logger.info("Loading from %s", load_from)
            state_dict = {
                f.stem: torch.load(f, 'cpu')
                for f in load_from.glob('*.pth')
            }
            self.runner.load_state_dict(state_dict, **self._load_state_dict)

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return self._checkpoint_dir

    @property
    def latest_checkpoint_dir(self) -> pathlib.Path:
        return self._latest_checkpoint_dir

    def _work_dir(self, name: str) -> pathlib.Path:
        return self._checkpoint_dir / name

    def _save(self, name: str) -> None:
        # for FSDP, all ranks should call state dict
        state_dict = self.runner.state_dict(**self._state_dict)

        if get_rank() != 0:
            return
        work_dir = self._work_dir(name)
        work_dir.mkdir(parents=True, exist_ok=True)
        self.runner.logger.info("Saving state dict to %s", work_dir)
        for k, v in state_dict.items():
            torch.save(v, work_dir / f'{k}.pth')

        if self._latest_checkpoint_dir.is_symlink():
            self._latest_checkpoint_dir.unlink()
        self._latest_checkpoint_dir.symlink_to(name)

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        if self._should_run_iter():
            self._save(f'iter_{self.runner.iter_}')

    def after_run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super().after_run_epoch(epoch_memo, memo)
        if self._should_run_epoch():
            self._save(f'epoch_{self.epoch_based_trainer.epoch}')

    # TODO: save the last checkpoint if not saved
