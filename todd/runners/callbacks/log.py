__all__ = [
    'LogCallback',
]

import datetime
import logging
from typing import Any, TypeVar

import torch
from torch import nn

from ...bases.configs import Config
from ...loggers import Formatter
from ...patches.torch import get_rank
from ...utils import Store, collect_env_, get_timestamp
from ..memo import Memo
from ..registries import CallbackRegistry
from ..utils import BaseETA, ETARegistry
from .base import BaseCallback
from .interval import IntervalMixin

T = TypeVar('T', bound=nn.Module)


@CallbackRegistry.register_()
class LogCallback(IntervalMixin[T], BaseCallback[T]):

    def __init__(
        self,
        *args,
        collect_env: Config | None = None,
        with_file_handler: bool = False,
        eta: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._collect_env = collect_env
        self._with_file_handler = with_file_handler
        self._eta_config = eta

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)

        if get_rank() > 0:
            return
        if self._with_file_handler:
            file = self.runner.work_dir / f'{get_timestamp()}.log'
            handler = logging.FileHandler(file)
            handler.setFormatter(Formatter())
            self.runner.logger.addHandler(handler)
        if self._collect_env is not None:
            env = collect_env_(**self._collect_env)
            self.runner.logger.info(env)

    def before_run(self, memo: Memo) -> None:
        super().before_run(memo)
        self._eta: BaseETA | None = (
            None if self._eta_config is None else ETARegistry.build(
                self._eta_config,
                start=self.runner.iter_ - 1,
                end=self.runner.iters,
            )
        )

    def before_run_iter(self, batch: Any, memo: Memo) -> None:
        super().before_run_iter(batch, memo)
        if get_rank() == 0 and self._should_run_iter():
            memo['log'] = dict()

    def after_run_iter(self, batch: Any, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        if 'log' not in memo:
            return
        prefix = f"Iter [{self.runner.iter_}/{self.runner.iters}] "

        if self._eta is not None:
            eta = self._eta(self.runner.iter_)
            eta = round(eta)
            prefix += f"ETA {str(datetime.timedelta(seconds=eta))} "

        if Store.cuda:  # pylint: disable=using-constant-test
            max_memory_allocated = max(
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            )
            torch.cuda.reset_peak_memory_stats()
            prefix += f"Memory {max_memory_allocated / 1024 ** 2:.2f}M "

        log: dict[str, Any] = memo.pop('log')
        message = ' '.join(f'{k}={v}' for k, v in log.items() if v is not None)
        self.runner.logger.info(prefix + message)

    def before_run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super().before_run_epoch(epoch_memo, memo)
        runner = self.epoch_based_trainer
        if get_rank() == 0:
            runner.logger.info(
                "Epoch [%d/%d]",
                runner.epoch + 1,
                runner.epochs,
            )
