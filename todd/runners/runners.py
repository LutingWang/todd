__all__ = [
    'BaseRunner',
    'Validator',
    'Trainer',
    'IterBasedTrainer',
    'EpochBasedTrainer',
]

import contextlib
import itertools
import logging
import os
import pathlib
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.distributed
import torch.utils.data

from ..base import (
    CallbackRegistry,
    Config,
    DatasetRegistry,
    OptimizerRegistry,
    SamplerRegistry,
    StateDict,
    StrategyRegistry,
)
from ..base import logger as base_logger

if TYPE_CHECKING:
    from .callbacks import BaseCallback
    from .strategies import BaseStrategy

Memo = dict[str, Any]

# TODO: split into multiple files


class BaseRunner(StateDict):

    def __init__(self, name: str, *args, **kwargs) -> None:
        self._name = name
        self._iter = 0
        self._build(*args, **kwargs)
        self._callbacks.connect(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def iter_(self) -> int:
        return self._iter

    @property
    def work_dir(self) -> pathlib.Path:
        return self._work_dir

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def strategy(self) -> 'BaseStrategy':
        return self._strategy

    @property
    def iters(self) -> int:
        return len(self._dataloader)

    @property
    def dataloader(self) -> torch.utils.data.DataLoader:
        return self._dataloader

    def _build_dataloader(self, *args, dataloader: Config, **kwargs) -> None:
        """Build the dataloader.

        Args:
            config: dataloader config.
        """
        dataloader = dataloader.copy()
        dataset = DatasetRegistry.build(dataloader.pop('dataset'))
        if 'sampler' in dataloader:
            dataloader.sampler = SamplerRegistry.build(
                dataloader.pop('sampler'),
                dataset=dataset,
            )
        self._dataloader = torch.utils.data.DataLoader(dataset, **dataloader)

    def _build_strategy(
        self,
        *args,
        strategy: Config,
        **kwargs,
    ) -> None:
        self._strategy: 'BaseStrategy' = StrategyRegistry.build(strategy)

    def _build_callbacks(
        self,
        *args,
        callbacks: Config | list[Config] | None = None,
        **kwargs,
    ) -> None:
        if callbacks is None:
            callbacks = []
        if isinstance(callbacks, list):
            callbacks = Config(type='ComposedCallback', callbacks=callbacks)
        self._callbacks: 'BaseCallback' = CallbackRegistry.build(callbacks)

    def _build_work_dir(
        self,
        *args,
        work_dir: Config | None = None,
        **kwargs,
    ) -> None:
        if work_dir is None:
            work_dir = Config()
        if 'path' in work_dir:
            path = work_dir.path
        else:
            root = work_dir.get('root', 'work_dirs')
            name = work_dir.get('name', self._name)
            path = os.path.join(root, name)
        self._work_dir = pathlib.Path(path)
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def _build_logger(
        self,
        *args,
        logger: Config | None = None,
        **kwargs,
    ) -> None:
        if logger is None:
            logger = Config()
        name = logger.get(
            'name',
            f'{self.__class__.__name__}.{self._name}',
        )
        self._logger = logging.getLogger(f'{base_logger.name}.{name}')

    def _build(self, *args, **kwargs) -> None:
        self._build_strategy(*args, **kwargs)
        self._build_dataloader(*args, **kwargs)
        self._build_callbacks(*args, **kwargs)
        self._build_work_dir(*args, **kwargs)
        self._build_logger(*args, **kwargs)

    @abstractmethod
    def _run_iter(self, batch, memo: Memo) -> None:
        pass

    def _run(self, memo: Memo) -> None:
        dataloader = memo['dataloader']
        for batch in dataloader:
            self._iter += 1

            if self._callbacks.should_break(self, batch, memo):
                break
            if self._callbacks.should_continue(self, batch, memo):
                continue

            self._callbacks.before_run_iter(self, batch, memo)
            with contextlib.ExitStack() as exit_stack:
                self._callbacks.run_iter_context(self, exit_stack, batch, memo)
                self._run_iter(batch, memo)
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
        super().load_state_dict(state_dict, *args, **kwargs)
        self._iter = state_dict['meta']['iter_']
        self._strategy.load_state_dict(
            state_dict['strategy'],
            *args,
            **kwargs,
        )
        self._callbacks.load_state_dict(
            state_dict['callbacks'],
            *args,
            **kwargs,
        )

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['meta'] = dict(iter_=self._iter)
        state_dict['strategy'] = self._strategy.state_dict(*args, **kwargs)
        state_dict['callbacks'] = self._callbacks.state_dict(*args, **kwargs)
        return state_dict

    def resume_from(self, f: pathlib.Path, *args, **kwargs) -> None:
        self._logger.info(f"Resuming from {f}")
        state_dict = torch.load(f, 'cpu')
        self.load_state_dict(state_dict, *args, **kwargs)

    def load_from(self, f: pathlib.Path, *args, **kwargs) -> None:
        self._logger.info(f"Loading from {f}")
        state_dict = torch.load(f, 'cpu')
        if 'strategy' not in state_dict:
            self._strategy.module.load_state_dict(state_dict, *args, **kwargs)
            return
        # runner checkpoint
        self._strategy.load_state_dict(
            state_dict['strategy'],
            *args,
            **kwargs,
        )


class Validator(BaseRunner):

    def _setup(self) -> Memo:
        self._strategy.model.eval()
        return super()._setup()

    @torch.no_grad()
    def run(self) -> Memo:
        return super().run()


class Trainer(BaseRunner):

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def _build_optimizer(
        self,
        *args,
        optimizer: Config,
        **kwargs,
    ) -> None:
        self._optimizer: torch.optim.Optimizer = OptimizerRegistry.build(
            optimizer,
            model=self._strategy.model,
        )

    def _build(self, *args, **kwargs) -> None:
        super()._build(*args, **kwargs)
        self._build_optimizer(*args, **kwargs)

    def _setup(self) -> Memo:
        self._strategy.model.train()
        return super()._setup()

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['optim'] = self._strategy.optim_state_dict(
            self,
            *args,
            **kwargs,
        )
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        self._strategy.load_optim_state_dict(
            self,
            state_dict['optim'],
            *args,
            **kwargs,
        )


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

    def _setup_epoch(self, memo: Memo) -> Memo:
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
            with contextlib.ExitStack() as exit_stack:
                self._callbacks.run_epoch_context(
                    self,
                    exit_stack,
                    epoch_memo,
                    memo,
                )
                self._run_epoch(epoch_memo, memo)
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
