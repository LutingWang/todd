__all__ = [
    'BaseRunner',
    'Validator',
    'Trainer',
    'IterBasedTrainer',
    'EpochBasedTrainer',
]

import getpass
import itertools
import logging
import pathlib
import socket
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.distributed
import torch.nn as nn
import torch.utils.data

from ..base import (
    CallbackRegistry,
    Config,
    LrSchedulerRegistry,
    OptimizerRegistry,
    Store,
    get_world_size,
    logger,
)

if TYPE_CHECKING:
    from .callbacks import BaseCallback
else:
    BaseCallback = object

Memo = dict[str, Any]


class BaseRunner(ABC):
    DRY_RUN_ITERS = 200

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        dataloader: Config,
        callbacks: Config,
        work_dirs_root: str = 'work_dirs',
    ) -> None:
        self._name = name
        self._model = model
        self._dataloader = self._build_dataloader(dataloader)
        self._callbacks = self._build_callbacks(callbacks)
        self._work_dir = self._build_work_dir(work_dirs_root, name)

        self._logger = self._build_logger()
        self._iter = 0

        self._logger.debug(
            f"Runner {name} initialized by {getpass.getuser()}@"
            f"{socket.gethostname()}"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> torch.nn.Module:
        """The underlying model if being wrapped."""
        model = self._model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        return model

    @property
    def work_dir(self) -> pathlib.Path:
        return self._work_dir

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def iter_(self) -> int:
        return self._iter

    @property
    def iters(self) -> int:
        return len(self._dataloader)

    @abstractmethod
    def _build_dataloader(self, config: Config) -> torch.utils.data.DataLoader:
        """Build the dataloader.

        Args:
            config: dataloader config.
        """
        pass

    def _build_callbacks(self, config: Config) -> BaseCallback:
        return CallbackRegistry.build(config)

    def _build_work_dir(self, root: str, name: str) -> pathlib.Path:
        work_dir = pathlib.Path(root) / name
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _build_logger(self) -> logging.Logger:
        name = f'{logger.name}.{self.__class__.__name__}.{self._name}'
        return logging.getLogger(name)

    @abstractmethod
    def _run_iter(self, batch, memo: Memo) -> None:
        pass

    def _run(self, memo: Memo) -> None:
        dataloader = memo['dataloader']
        for batch in dataloader:
            self._iter += 1
            if Store.DRY_RUN and self._iter > self.DRY_RUN_ITERS:
                break
            if self._callbacks.should_break(self, batch, memo):
                break
            if self._callbacks.should_continue(self, batch, memo):
                continue

            self._callbacks.before_run_iter(self, batch, memo)
            try:
                self._run_iter(batch, memo)
            except Exception:
                self._logger.exception(
                    f"Unable to run iter {self._iter}\n{batch=}\n{memo=}"
                )
                raise
            self._callbacks.after_run_iter(self, batch, memo)

    def _setup(self) -> Memo:
        return dict(dataloader=self._dataloader)

    def _teardown(self, memo: Memo) -> None:
        pass

    def run(self) -> Memo:
        memo = self._setup()
        self._callbacks.before_run(self, memo)
        self._run(memo)
        self._callbacks.after_run(self, memo)
        self._teardown(memo)
        return memo

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        self._iter = state_dict['meta']['iter_']
        self.model.load_state_dict(state_dict['model'], *args, **kwargs)

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        meta = dict(iter_=self._iter)
        model = self.model.state_dict(*args, **kwargs)
        return dict(meta=meta, model=model)

    def load_state_dict_from(self, f: pathlib.Path, *args, **kwargs) -> None:
        self._logger.info(f"Loading state dict from {f}")
        state_dict = torch.load(f, 'cpu')
        self.load_state_dict(state_dict, *args, **kwargs)


class Validator(BaseRunner):

    def _setup(self) -> Memo:
        self._model.eval()
        return super()._setup()

    def _teardown(self, memo: Memo) -> None:
        super()._teardown(memo)
        self._model.train()

    @torch.no_grad()
    def run(self) -> Memo:
        return super().run()


class Trainer(BaseRunner):

    def __init__(
        self,
        *args,
        optimizer: Config,
        lr_scheduler: Config | None = None,
        lr_scalar: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._optimizer = self._build_optimizer(optimizer, self.model)
        if lr_scalar is not None:
            self._lr_scalar = self._build_lr_scalar(lr_scalar)
            self._scale_lr(self._optimizer, self._lr_scalar)
        if lr_scheduler is not None:
            self._lr_scheduler = self._build_lr_scheduler(
                lr_scheduler, self._optimizer
            )

    @property
    def with_lr_scheduler(self) -> bool:
        return hasattr(self, '_lr_scheduler')

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self._lr_scheduler

    def _build_optimizer(
        self,
        config: Config,
        model: nn.Module,
    ) -> torch.optim.Optimizer:
        return OptimizerRegistry.build(config, Config(model=model))

    def _build_lr_scheduler(
        self,
        config: Config,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return LrSchedulerRegistry.build(config, Config(optimizer=optimizer))

    def _build_lr_scalar(self, config: Config) -> float:
        base_batch_size = config.base_batch_size
        batch_size = get_world_size() * self._dataloader.batch_size
        return batch_size / base_batch_size

    def _scale_lr(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scalar: float,
    ) -> None:
        if 'lr' in optimizer.defaults:
            self._optimizer.defaults['lr'] *= lr_scalar
        for param_group in optimizer.param_groups:
            if 'lr' in param_group:
                param_group['lr'] *= lr_scalar

    def _setup(self) -> Memo:
        self._model.train()
        return super()._setup()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        self._optimizer.load_state_dict(state_dict['optimizer'])
        if self.with_lr_scheduler:
            self._lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['optimizer'] = self._optimizer.state_dict()
        if self.with_lr_scheduler:
            state_dict['lr_scheduler'] = self._lr_scheduler.state_dict()
        return state_dict


class IterBasedTrainer(Trainer):

    def __init__(self, *args, iters: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._iters = iters

    @property
    def iters(self) -> int:
        return self._iters

    def _setup(self) -> Memo:
        memo = super()._setup()
        dataloader = memo['dataloader']
        dataloader = itertools.cycle(dataloader)
        dataloader = itertools.islice(dataloader, self._iters)
        memo['dataloader'] = dataloader
        return memo


class EpochBasedTrainer(Trainer):

    def __init__(self, *args, epochs: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._epoch = 0
        self._epochs = epochs

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def iters(self) -> int:
        return super().iters * self._epochs

    @property
    def epochs(self) -> int:
        return self._epochs

    def _run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super()._run(epoch_memo)

    def _setup_epoch(self, memo) -> Memo:
        sampler = self._dataloader.batch_sampler
        if sampler is None:
            sampler = self._dataloader.sampler
        if isinstance(sampler, torch.utils.data.DistributedSampler):
            sampler.set_epoch(self._epoch)
        return super()._setup()

    def _teardown_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super()._teardown(epoch_memo)
        memo['epoch_memos'][self._epoch] = epoch_memo

    def _run(self, memo: Memo) -> None:
        while self._epoch < self._epochs:
            self._epoch += 1
            epoch_memo = self._setup_epoch(memo)

            if self._callbacks.should_break_epoch(self, epoch_memo, memo):
                break
            if self._callbacks.should_continue_epoch(self, epoch_memo, memo):
                continue

            self._callbacks.before_run_epoch(self, epoch_memo, memo)
            try:
                self._run_epoch(epoch_memo, memo)
            except Exception:
                self._logger.exception(
                    f"Unable to run epoch {self._epoch}\n{epoch_memo=}\n"
                    f"{memo=}"
                )
                raise
            self._callbacks.after_run_epoch(self, epoch_memo, memo)

            self._teardown_epoch(epoch_memo, memo)

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
