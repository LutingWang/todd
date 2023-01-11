__all__ = [
    'BaseRunner',
    'Validator',
    'Trainer',
    'IterBasedTrainer',
    'EpochBasedTrainer',
    'RunnerRegistry',
    'Memo',
]

import pathlib
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
import torch.distributed
import torch.utils.data

from ..base import (
    Config,
    LrSchedulerRegistry,
    OptimizerRegistry,
    Registry,
    StoreMeta,
    get_logger,
    get_rank,
    get_world_size,
)

Memo = dict[str, Any]


@runtime_checkable
class DistributedSamplerProto(Protocol):

    def set_epoch(self, epoch: int) -> None:
        ...


class BaseRunner(ABC):

    class Store(metaclass=StoreMeta):
        CPU: bool
        CUDA: bool

        DRY_RUN: bool
        TRAIN_WITH_VAL_DATASET: bool

    if not Store.CPU and not Store.CUDA:
        if torch.cuda.is_available():
            Store.CUDA = True
        else:
            Store.CPU = True
    assert Store.CPU + Store.CUDA == 1

    if Store.CPU:
        Store.DRY_RUN = True
        Store.TRAIN_WITH_VAL_DATASET = True

        try:
            from mmcv.cnn import NORM_LAYERS
            NORM_LAYERS.register_module(
                name='SyncBN',
                force=True,
                module=torch.nn.BatchNorm2d,
            )
            del NORM_LAYERS
        except Exception:
            pass

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        dataloader: Config,
        log: Config,
        load_state_dict: Config,
        state_dict: Config,
    ) -> None:
        work_dir = pathlib.Path('work_dirs') / name
        work_dir.mkdir(parents=True, exist_ok=True)
        self._work_dir = work_dir

        log_file = time.strftime('%Y%m%dT%H%M%S%f') + '.log'
        self._logger = get_logger(id(self), work_dir / log_file)

        self._model = model

        self._dataloader = self._build_dataloader(dataloader)
        self._log = log
        self._load_state_dict = load_state_dict
        self._state_dict = state_dict

    @property
    def model(self) -> torch.nn.Module:
        """The underlying model if being wrapped."""
        model = self._model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        return model

    @abstractmethod
    def _build_dataloader(self, config: Config) -> torch.utils.data.DataLoader:
        """Build the dataloader.

        Args:
            config: dataloader config.
        """
        pass

    def _stop_run_iter(self, i: int, batch, memo: Memo) -> bool:
        """Whether the current iteration should execute or not.

        Args:
            i: iteration index.
            batch: inputs.
            memo: runtime memory.

        Override this method for early stopping, error detection, etc.
        By default, this method supports `DRY_RUN` by returning `True` after
        the first log message.
        """
        return self.Store.DRY_RUN and i > self._log.interval

    def _before_run_iter(self, i: int, batch, memo: Memo) -> None:
        pass

    def _before_run_iter_log(self, i: int, batch, memo: Memo) -> str | None:
        pass

    @abstractmethod
    def _run_iter(self, i: int, batch, memo: Memo) -> torch.Tensor:
        pass

    def _after_run_iter(self, i: int, batch, memo: Memo) -> None:
        pass

    def _after_run_iter_log(self, i: int, batch, memo: Memo) -> str | None:
        pass

    def _before_run(self) -> Memo:
        return dict()

    def _run(self, memo: Memo) -> None:
        for i, batch in enumerate(self._dataloader, 1):
            if self._stop_run_iter(i, batch, memo):
                return
            log = i % self._log.interval == 0
            self._before_run_iter(i, batch, memo)
            if log:
                info = self._before_run_iter_log(i, batch, memo)
                if info is not None:
                    self._logger.info(info)
            try:
                memo['loss'] = self._run_iter(i, batch, memo)
            except Exception:
                self._logger.exception(
                    f"Unable to run iter {i}\n{batch=}\n{memo=}"
                )
                raise
            self._after_run_iter(i, batch, memo)
            if log:
                info = self._after_run_iter_log(i, batch, memo)
                if info is not None:
                    self._logger.info(info)

    def _after_run(self, memo: Memo) -> None:
        pass

    def run(self) -> Memo:
        memo = self._before_run()
        self._run(memo)
        self._after_run(memo)
        return memo

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.model.load_state_dict(
            state_dict['model'],
            **self._load_state_dict.model,
        )

    def state_dict(self) -> dict[str, Any]:
        model = self.model.state_dict(**self._state_dict.model)
        return dict(model=model)

    def write_state_dict(self, f: pathlib.Path) -> None:
        self._logger.info(f"Writing state dict to {f}")
        torch.save(self.state_dict(), f)

    def read_state_dict(self, f: pathlib.Path) -> None:
        self._logger.info(f"Reading state dict from {f}")
        self.load_state_dict(torch.load(f, 'cpu'))


class Validator(BaseRunner):

    def _after_run_iter_log(self, i: int, batch, memo: Memo) -> str | None:
        info = f"Iter [{i}/{len(self._dataloader)}]"
        loss: torch.Tensor = memo['loss']
        info += f" Loss {loss.item():.3f}"
        return info

    def _before_run(self) -> Memo:
        self._model.eval()
        return super()._before_run()

    @torch.no_grad()
    def run(self) -> Memo:
        return super().run()


class Trainer(BaseRunner):

    def __init__(
        self,
        *args,
        optimizer: Config,
        lr_scheduler: Config | None = None,
        validator: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._optimizer = self._build_optimizer(optimizer)
        self._lr_scheduler = (
            None if lr_scheduler is None else
            self._build_lr_scheduler(lr_scheduler)
        )
        self._validator: Validator | None = (
            None if validator is None else RunnerRegistry.build(validator)
        )

    def _build_optimizer(self, config: Config) -> torch.optim.Optimizer:
        return OptimizerRegistry.build(config, Config(model=self.model))

    def _build_lr_scheduler(
        self,
        config: Config,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return LrSchedulerRegistry.build(config)

    def _after_run_iter(self, i: int, batch, memo: Memo) -> None:
        loss: torch.Tensor = memo['loss']
        loss.backward()
        self._optimizer.step()
        self._model.zero_grad()

    def _after_run_iter_log(self, i: int, batch, memo: Memo) -> str | None:
        if self._lr_scheduler is None:
            info = ""
        else:
            info = "LR "
            info += str([
                f'{lr:.3e}' for lr in self._lr_scheduler.get_last_lr()
            ])
            info += " "
        loss: torch.Tensor = memo['loss']
        info += f"Loss {loss.item():.3f}"
        return info

    def _before_run(self) -> Memo:
        self._model.train()
        return super()._before_run()

    def _after_run(self, memo: Memo) -> None:
        self.validate()

    def validate(self) -> None:
        if self._validator is None:
            self._logger.info("Skipping validation")
            return
        if get_world_size() > 1:
            torch.distributed.barrier()
        self._validator.run()
        self._model.train()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._optimizer.load_state_dict(
            state_dict['optimizer'],
            **self._load_state_dict.optimizer,
        )
        if self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(
                state_dict['lr_scheduler'],
                **self._load_state_dict.lr_scheduler,
            )

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict['optimizer'] = self._optimizer.state_dict(
            **self._state_dict.optimizer,
        )
        if self._lr_scheduler is not None:
            state_dict['lr_scheduler'] = self._lr_scheduler.state_dict(
                **self._state_dict.lr_scheduler,
            )
        return state_dict


class IterBasedTrainer(Trainer):

    def __init__(self, *args, iters: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._iter = 1
        self._iters = iters

    def _stop_run_iter(self, i: int, batch, memo: Memo) -> bool:
        return (
            self._iter > self._iters or super()._stop_run_iter(i, batch, memo)
        )

    def _after_run_iter(self, i: int, batch, memo: Memo) -> None:
        super()._after_run_iter(i, batch, memo)
        if self._iter % self._state_dict.interval == 0:
            self.validate()
            self.write_state_dict(self._work_dir / f'iter_{self._iter}.pth')
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        self._iter += 1

    def _after_run_iter_log(self, i: int, batch, memo: Memo) -> str | None:
        info = super()._after_run_iter_log(i, batch, memo)
        info = "" if info is None else " " + info
        info = f"Iter [{self._iter}/{self._iters}]" + info
        return info

    def _run(self, memo: Memo) -> None:
        while self._iter <= self._iters:
            super()._run(memo)

    def _after_run(self, memo: Memo):
        self.write_state_dict(self._work_dir / 'latest.pth')
        return super()._after_run(memo)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._iter = state_dict['iter_']

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict['iter_'] = self._iter
        return state_dict


class EpochBasedTrainer(Trainer):

    def __init__(self, *args, epochs: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._epoch = 1
        self._epochs = epochs

    def _build_lr_scheduler(
        self,
        config: Config,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        self._lr_scheduler_by_epoch = config.pop('by_epoch', True)
        return super()._build_lr_scheduler(config)

    def _after_run_iter(self, i: int, batch, memo: Memo) -> None:
        super()._after_run_iter(i, batch, memo)
        if self._lr_scheduler is not None and not self._lr_scheduler_by_epoch:
            self._lr_scheduler.step()

    def _after_run_iter_log(self, i: int, batch, memo: Memo) -> str | None:
        info = super()._after_run_iter_log(i, batch, memo)
        info = "" if info is None else " " + info
        info = f"Iter [{i}/{len(self._dataloader)}]" + info
        return info

    def _stop_run_epoch(self, memo: Memo) -> bool:
        return False

    def _before_run_epoch(self, memo: Memo) -> Memo:
        sampler = self._dataloader.batch_sampler
        if sampler is None:
            sampler = self._dataloader.sampler
        if isinstance(sampler, DistributedSamplerProto):
            sampler.set_epoch(self._epoch)
        return super()._before_run()

    def _before_run_epoch_log(
        self,
        epoch_memo: Memo,
        memo: Memo,
    ) -> str | None:
        return f"Epoch [{self._epoch}/{self._epochs}] beginning"

    def _run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super()._run(epoch_memo)

    def _after_run_epoch(
        self,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        memo['epoch_memos'][self._epoch] = epoch_memo
        self.write_state_dict(self._work_dir / f'epoch_{self._epoch}.pth')
        if self._lr_scheduler is not None and self._lr_scheduler_by_epoch:
            self._lr_scheduler.step()
        self._epoch += 1

    def _after_run_epoch_log(
        self,
        epoch_memo: Memo,
        memo: Memo,
    ) -> str | None:
        return f"Epoch [{self._epoch}/{self._epochs}] ended"

    def _before_run(self) -> Memo:
        memo: Memo = dict(epoch_memos=dict())
        return memo

    def _run(self, memo: Memo) -> None:
        while self._epoch <= self._epochs:
            if self._stop_run_epoch(memo):
                return
            log = get_rank() == 0
            epoch_memo = self._before_run_epoch(memo)
            if log:
                info = self._before_run_epoch_log(epoch_memo, memo)
                if info is not None:
                    self._logger.info(info)
            try:
                self._run_epoch(epoch_memo, memo)
            except Exception:
                self._logger.exception(
                    f"Unable to run epoch {self._epoch}\n{epoch_memo=}\n"
                    f"{memo=}"
                )
                raise
            self._after_run_epoch(epoch_memo, memo)
            if get_rank() == 0:
                info = self._after_run_epoch_log(epoch_memo, memo)
                if info is not None:
                    self._logger.info(info)

    def _after_run(self, memo: Memo):
        pass


class RunnerRegistry(Registry):
    pass
