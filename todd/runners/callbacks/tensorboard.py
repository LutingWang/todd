__all__ = [
    'TensorBoardCallback',
]

import pathlib
from typing import Any, TypeVar

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ...bases.configs import Config
from ...patches.torch import get_rank
from ..memo import Memo
from ..registries import CallbackRegistry
from .base import BaseCallback
from .interval import IntervalMixin

T = TypeVar('T', bound=nn.Module)


@CallbackRegistry.register_()
class TensorBoardCallback(IntervalMixin[T], BaseCallback[T]):

    def __init__(
        self,
        *args,
        summary_writer: Config | None = None,
        main_tag: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if summary_writer is None:
            summary_writer = Config()
        self._summary_writer_config = summary_writer
        self._main_tag = main_tag

    @property
    def work_dir(self) -> pathlib.Path:
        return self.runner.work_dir / 'tensorboard'

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)

        if get_rank() > 0:
            return
        self._summary_writer = SummaryWriter(
            self.work_dir,
            **self._summary_writer_config,
        )

    @property
    def summary_writer(self) -> SummaryWriter:
        return self._summary_writer

    @property
    def main_tag(self) -> str:
        return self._main_tag

    def tag(self, tag: str) -> str:
        assert tag
        return self._main_tag + '/' + tag

    def before_run_iter(self, batch: Any, memo: Memo) -> None:
        super().before_run_iter(batch, memo)
        if get_rank() == 0 and self._should_run_iter():
            memo['tensorboard'] = self

    def after_run_iter(self, batch: Any, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        memo.pop('tensorboard', None)
