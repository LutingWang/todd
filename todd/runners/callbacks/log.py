__all__ = [
    'LogCallback',
]

import datetime
import logging
from typing import Any

import torch

from ...base import CallbackRegistry, Config, EnvRegistry, Formatter, Store
from ...utils import get_rank, get_timestamp
from .base import BaseCallback
from .interval import IntervalMixin

Memo = dict[str, Any]


@CallbackRegistry.register()
class LogCallback(IntervalMixin, BaseCallback):

    def __init__(
        self,
        *args,
        collect_env: Config | None = None,
        with_file_handler: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._collect_env = collect_env
        self._with_file_handler = with_file_handler

    def init(self) -> None:
        super().init()
        if get_rank() > 0 or not self._with_file_handler:
            return
        file = self._runner.work_dir / f'{get_timestamp()}.log'
        handler = logging.FileHandler(file)
        handler.setFormatter(Formatter())
        self._runner.logger.addHandler(handler)
        if self._collect_env is None:
            return
        envs = ['']
        for k, v in EnvRegistry.items():
            env = str(v(**self._collect_env))
            env = env.strip()
            if '\n' in env:
                env = '\n' + env
            envs.append(f'{k}: {env}')
        self._runner.logger.info('\n'.join(envs))

    def before_run(self, memo: Memo) -> None:
        super().before_run(memo)
        self._start_time = datetime.datetime.now()
        self._start_iter = self._runner.iter_

    def before_run_iter(self, batch, memo: Memo) -> None:
        super().before_run_iter(batch, memo)
        if get_rank() == 0 and self._should_run_iter():
            memo['log'] = dict()

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        if 'log' not in memo:
            return
        prefix = f"Iter [{self._runner.iter_}/{self._runner.iters}] "

        eta = datetime.datetime.now() - self._start_time
        eta *= self._runner.iters - self._runner.iter_
        eta /= self._runner.iter_ - self._start_iter
        eta = datetime.timedelta(seconds=round(eta.total_seconds()))
        prefix += f"ETA {str(eta)} "

        if Store.CUDA:
            max_memory_allocated = max(
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            )
            torch.cuda.reset_peak_memory_stats()
            prefix += f"Memory {max_memory_allocated / 1024 ** 2:.2f}M "

        log: dict[str, Any] = memo.pop('log')
        message = ' '.join(f'{k}={v}' for k, v in log.items())
        self._runner.logger.info(prefix + message)

    def before_run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super().before_run_epoch(epoch_memo, memo)
        runner = self.epoch_based_trainer
        if get_rank() == 0:
            runner.logger.info(f"Epoch [{runner.epoch}/{runner.epochs}]")
