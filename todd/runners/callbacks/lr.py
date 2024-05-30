__all__ = [
    'LRScheduleCallback',
    'LRScaleCallback',
]

from typing import Any, Mapping, cast

import torch

from ...configs import Config
from ...patches.torch import get_rank, get_world_size
from ...registries import LRSchedulerRegistry
from ..memo import Memo
from ..registries import CallbackRegistry
from ..trainer import Trainer
from .base import BaseCallback
from .interval import IntervalMixin


@CallbackRegistry.register_()
class LRScheduleCallback(IntervalMixin, BaseCallback):

    def __init__(
        self,
        *args,
        lr_scheduler: Config,
        interval: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, interval=interval, **kwargs)
        assert isinstance(self.runner, Trainer)
        self._lr_scheduler_config = lr_scheduler

    def init(self, *args, **kwargs) -> None:
        super().init(*args, **kwargs)
        self._build_lr_scheduler()

    def _build_lr_scheduler(self) -> None:
        runner = cast(Trainer, self.runner)
        self._lr_scheduler: torch.optim.lr_scheduler.LRScheduler = \
            LRSchedulerRegistry.build(
                self._lr_scheduler_config,
                optimizer=runner.optimizer,
            )

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        if 'log' in memo:
            memo['log']['lr'] = [
                f'{lr:.3e}' for lr in self._lr_scheduler.get_last_lr()
            ]
        if self._should_run_iter():
            self._lr_scheduler.step()

    def after_run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super().after_run_epoch(epoch_memo, memo)
        if self._should_run_epoch():
            self._lr_scheduler.step()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        self._lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['lr_scheduler'] = self._lr_scheduler.state_dict()
        return state_dict


@CallbackRegistry.register_()
class LRScaleCallback(BaseCallback):

    def __init__(self, *args, lr_scaler: Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.runner, Trainer)
        self._lr_scaler_config = lr_scaler

    def _scale_lr(self, config: Config) -> None:
        runner = cast(Trainer, self.runner)
        assert runner.dataloader.batch_size is not None
        base_batch_size = config.base_batch_size
        batch_size = get_world_size() * runner.dataloader.batch_size
        lr_scaler = batch_size / base_batch_size
        if 'lr' in runner.optimizer.defaults:
            runner.optimizer.defaults['lr'] *= lr_scaler
        for param_group in runner.optimizer.param_groups:
            if 'lr' in param_group:
                param_group['lr'] *= lr_scaler
        if get_rank() == 0:
            runner.logger.info(
                f"{base_batch_size=} {batch_size=} {lr_scaler=:.3f}"
            )

    def init(self, *args, **kwargs) -> None:
        super().init(*args, **kwargs)
        self._scale_lr(self._lr_scaler_config)
