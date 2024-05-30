__all__ = [
    'BaseRunner',
]

import contextlib
import getpass
import logging
import os
import pathlib
import socket
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Iterable, Mapping, TypeVar

from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..configs import Config
from ..loggers import logger as base_logger
from ..patches.torch import get_rank
from ..registries import (
    CollateRegistry,
    DatasetRegistry,
    RunnerRegistry,
    SamplerRegistry,
)
from ..utils import StateDictMixin
from .memo import Memo
from .registries import StrategyRegistry

if TYPE_CHECKING:
    from .callbacks import ComposedCallback
    from .strategies import BaseStrategy

T = TypeVar('T', bound=nn.Module)


@RunnerRegistry.register_()
class BaseRunner(StateDictMixin, Generic[T]):

    def __init__(
        self,
        name: str,
        *args,
        load_from: str | None = None,
        auto_resume: bool = False,
        **kwargs,
    ) -> None:
        self._name = name
        self._load_from = load_from
        self._auto_resume = auto_resume

        self._iter = 0
        self._build(*args, **kwargs)
        self._init_callbacks()

        self._logger.debug(
            "Rank %d initialized by %s@%s",
            get_rank(),
            getpass.getuser(),
            socket.gethostname(),
        )

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} name={self._name} "
            f"load_from={self._load_from} auto_resume={self._auto_resume}>"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def load_from(self) -> str | None:
        return self._load_from

    @property
    def iter_(self) -> int:
        return self._iter

    @property
    def strategy(self) -> 'BaseStrategy[T]':
        return self._strategy

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def model(self) -> T:
        return self._model

    @property
    def dataloader(self) -> DataLoader:
        return self._dataloader

    @property
    def callbacks(self) -> 'ComposedCallback':
        return self._callbacks

    @property
    def work_dir(self) -> pathlib.Path:
        return self._work_dir

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    @abstractmethod
    def iters(self) -> int:
        pass

    def _build_strategy(
        self,
        *args,
        strategy: Config,
        **kwargs,
    ) -> None:
        self._strategy: 'BaseStrategy[T]' = StrategyRegistry.build(
            strategy,
            runner=self,
        )

    def _build_dataset(self, *args, dataset: Config, **kwargs) -> None:
        self._dataset: Dataset = DatasetRegistry.build(dataset)

    def _build_model(
        self,
        *args,
        model: Config,
        map_model: Config | None = None,
        wrap_model: Config | None = None,
        **kwargs,
    ) -> None:
        if map_model is None:
            map_model = Config()
        if wrap_model is None:
            wrap_model = Config()
        model_ = self._strategy.build_model(model)
        model_ = self._strategy.map_model(model_, map_model)
        model_ = self._strategy.wrap_model(model_, wrap_model)
        self._model = model_

    def _build_dataloader(self, *args, dataloader: Config, **kwargs) -> None:
        """Build the dataloader.

        Args:
            dataloader: dataloader config.
        """
        dataloader = dataloader.copy()

        sampler = dataloader.pop('sampler', None)
        if sampler is not None:
            dataloader.sampler = SamplerRegistry.build(
                sampler,
                dataset=self._dataset,
            )

        batch_sampler = dataloader.pop('batch_sampler', None)
        if batch_sampler is not None:
            dataloader.batch_sampler = SamplerRegistry.build(
                batch_sampler,
                sampler=dataloader.pop('sampler'),
            )

        collate_fn = dataloader.pop('collate_fn', None)
        if collate_fn is not None:
            dataloader.collate_fn = CollateRegistry.build(collate_fn)

        self._dataloader = DataLoader(self._dataset, **dataloader)

    def _build_callbacks(
        self,
        *args,
        callbacks: Iterable[Config] | None = None,
        **kwargs,
    ) -> None:
        from .callbacks import ComposedCallback
        if callbacks is None:
            callbacks = []
        self._callbacks = ComposedCallback(runner=self, callbacks=callbacks)

    def _build_work_dir(
        self,
        *args,
        work_dir: Config | None = None,
        **kwargs,
    ) -> None:
        if work_dir is None:
            work_dir = Config()
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
        self._build_dataset(*args, **kwargs)
        self._build_model(*args, **kwargs)
        self._build_dataloader(*args, **kwargs)
        self._build_callbacks(*args, **kwargs)
        self._build_work_dir(*args, **kwargs)
        self._build_logger(*args, **kwargs)

    def _init_callbacks(self) -> None:
        self._callbacks.init()

    def _run_iter(self, batch, memo: Memo, *args, **kwargs) -> Memo:
        """Run iteration.

        Args:
            batch: input data
            memo: runtime memory

        Returns:
            Updated runtime memory.
        """
        return self._model(self, batch, memo, *args, **kwargs)

    def _run(self, memo: Memo) -> Memo:
        dataloader = memo['dataloader']
        for batch in dataloader:
            self._iter += 1

            if self._callbacks.should_break(batch, memo):
                break
            if self._callbacks.should_continue(batch, memo):
                continue

            self._callbacks.before_run_iter(batch, memo)
            with contextlib.ExitStack() as exit_stack:
                self._callbacks.run_iter_context(exit_stack, batch, memo)
                memo = self._run_iter(batch, memo)
            self._callbacks.after_run_iter(batch, memo)
        return memo

    def _setup(self) -> Memo:
        return dict()

    def _teardown(self, memo: Memo) -> None:
        pass

    def run(self) -> Memo:
        memo = self._setup()
        self._callbacks.before_run(memo)
        memo = self._run(memo)
        self._callbacks.after_run(memo)
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
        self._strategy.load_model_state_dict(
            state_dict['model'],
            *args,
            **kwargs,
        )
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
        state_dict['model'] = self._strategy.model_state_dict(*args, **kwargs)
        state_dict['strategy'] = self._strategy.state_dict(*args, **kwargs)
        state_dict['callbacks'] = self._callbacks.state_dict(*args, **kwargs)
        return state_dict
