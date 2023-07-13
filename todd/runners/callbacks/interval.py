__all__ = [
    'IntervalMixin',
]

from ..runners import BaseRunner, EpochBasedTrainer
from .base import BaseCallback


class IntervalMixin(BaseCallback):

    def __init__(
        self,
        *args,
        interval: int = 0,
        by_epoch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._interval = interval
        self._by_epoch = by_epoch

    def __should_run(self, step: int) -> bool:
        return self._interval > 0 and step % self._interval == 0

    def _should_run_iter(self, runner: BaseRunner) -> bool:
        return not self._by_epoch and self.__should_run(runner.iter_)

    def _should_run_epoch(self, runner: EpochBasedTrainer) -> bool:
        return self._by_epoch and self.__should_run(runner.epoch)
