__all__ = [
    'CheckpointCallback',
]

import pathlib
from typing import TypeVar

import torch
from torch import nn

from ...bases.configs import Config
from ...patches.torch import get_rank
from ..memo import Memo
from ..registries import CallbackRegistry
from .base import BaseCallback
from .interval import IntervalMixin

T = TypeVar('T', bound=nn.Module)


@CallbackRegistry.register_()
class CheckpointCallback(IntervalMixin[T], BaseCallback[T]):

    def __init__(
        self,
        *args,
        state_dict: Config | None = None,
        load_state_dict: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if state_dict is None:
            state_dict = Config()
        self._state_dict = state_dict
        if load_state_dict is None:
            load_state_dict = Config()
        self._load_state_dict = load_state_dict

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)

        self.work_dir.mkdir(parents=True, exist_ok=True)

        if self.runner.auto_resume and self.latest_checkpoint_dir.exists():
            load_from = self.latest_checkpoint_dir
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
    def work_dir(self) -> pathlib.Path:
        return self.runner.work_dir / 'checkpoints'

    @property
    def latest_checkpoint_dir(self) -> pathlib.Path:
        return self._checkpoint_dir('latest')

    def _checkpoint_dir(self, name: str) -> pathlib.Path:
        return self.work_dir / name

    def _save(self, name: str) -> None:
        # for FSDP, all ranks should call state dict
        state_dict = self.runner.state_dict(**self._state_dict)

        if get_rank() != 0:
            return
        checkpoint_dir = self._checkpoint_dir(name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.runner.logger.info("Saving state dict to %s", checkpoint_dir)
        for k, v in state_dict.items():
            torch.save(v, checkpoint_dir / f'{k}.pth')

        self.latest_checkpoint_dir.unlink(True)
        self.latest_checkpoint_dir.symlink_to(checkpoint_dir.absolute(), True)

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        if self._should_run_iter():
            self._save(f'iter_{self.runner.iter_}')

    def after_run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super().after_run_epoch(epoch_memo, memo)
        if self._should_run_epoch():
            self._save(f'epoch_{self.epoch_based_trainer.epoch}')
