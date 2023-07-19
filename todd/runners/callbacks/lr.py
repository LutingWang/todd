__all__ = [
    'LRScheduleCallback',
    'LRScaleCallback',
]

from typing import Any, Mapping

import torch

from ...base import CallbackRegistry, Config, LrSchedulerRegistry
from ...utils import get_world_size
from ..runners import BaseRunner, EpochBasedTrainer, Trainer
from .base import BaseCallback
from .interval import IntervalMixin

Memo = dict[str, Any]


@CallbackRegistry.register()
class LRScheduleCallback(IntervalMixin, BaseCallback):

    def __init__(
        self,
        *args,
        lr_scheduler: Config,
        interval: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, interval=interval, **kwargs)
        self._lr_scheduler_config = lr_scheduler

    def connect(self, runner: BaseRunner) -> None:
        assert isinstance(runner, Trainer)
        super().connect(runner)
        self._build_lr_scheduler(runner)

    def _build_lr_scheduler(self, runner: Trainer) -> None:
        self._lr_scheduler: torch.optim.lr_scheduler.LRScheduler = \
            LrSchedulerRegistry.build(
                self._lr_scheduler_config,
                optimizer=runner.optimizer,
            )

    def after_run_iter(self, runner: BaseRunner, batch, memo: Memo) -> None:
        assert isinstance(runner, Trainer)
        if self._should_run_iter(runner):
            self._lr_scheduler.step()
        if 'log' in memo:
            memo['log']['lr'] = [
                f'{lr:.3e}' for lr in self._lr_scheduler.get_last_lr()
            ]

    def after_run_epoch(
        self,
        runner: EpochBasedTrainer,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        if self._should_run_epoch(runner):
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


@CallbackRegistry.register()
class LRScaleCallback(BaseCallback):

    def __init__(self, *args, lr_scaler: Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lr_scaler_config = lr_scaler

    def _scale_lr(self, runner: Trainer, config: Config) -> None:
        base_batch_size = config.base_batch_size
        batch_size = get_world_size() * runner.dataloader.batch_size
        lr_scaler = batch_size / base_batch_size
        if 'lr' in runner.optimizer.defaults:
            runner.optimizer.defaults['lr'] *= lr_scaler
        for param_group in runner.optimizer.param_groups:
            if 'lr' in param_group:
                param_group['lr'] *= lr_scaler
        runner.logger.info(
            f"{base_batch_size=} {batch_size=} {lr_scaler=:.3f}"
        )

    def connect(self, runner: BaseRunner) -> None:
        assert isinstance(runner, Trainer)
        super().connect(runner)
        self._scale_lr(runner, self._lr_scaler_config)
