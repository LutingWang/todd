__all__ = [
    'IntervalMixin',
]

from ..runners import BaseRunner, EpochBasedTrainer
from .base import BaseCallback


class IntervalMixin(BaseCallback):

    def __init__(self, *args, interval: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._interval = interval

    def __should_run(self, step: int) -> bool:
        return self._interval > 0 and step % self._interval == 0

    def _should_run_iter(self, runner: BaseRunner) -> bool:
        return self.__should_run(runner.iter_)

    def _should_run_epoch(self, runner: EpochBasedTrainer) -> bool:
        return self.__should_run(runner.epoch)
