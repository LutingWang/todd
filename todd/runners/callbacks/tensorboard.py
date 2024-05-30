__all__ = [
    'TensorBoardCallback',
]

from torch.utils.tensorboard import SummaryWriter

from ...configs import Config
from ...patches.torch import get_rank
from ..memo import Memo
from ..registries import CallbackRegistry
from .base import BaseCallback
from .interval import IntervalMixin


@CallbackRegistry.register_()
class TensorBoardCallback(IntervalMixin, BaseCallback):

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

    def init(self, *args, **kwargs) -> None:
        super().init(*args, **kwargs)
        if get_rank() > 0:
            return
        log_dir = self.runner.work_dir / 'tensorboard'
        self._summary_writer = SummaryWriter(
            log_dir,
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

    def before_run_iter(self, batch, memo: Memo) -> None:
        super().before_run_iter(batch, memo)
        if get_rank() == 0 and self._should_run_iter():
            memo['tensorboard'] = self

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        memo.pop('tensorboard', None)
