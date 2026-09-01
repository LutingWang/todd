__all__ = [
    'IntervalMixin',
]

from typing import TypeVar

from torch import nn

from ...utils import Store
from .base import BaseCallback

T = TypeVar('T', bound=nn.Module)


class IntervalMixin(BaseCallback[T]):

    def __init__(
        self,
        *args,
        interval: int = 0,
        by_epoch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if Store.DRY_RUN:
            interval = 1

        self._interval = interval
        self._by_epoch = by_epoch

    def __should_run(self, step: int) -> bool:
        return self._interval > 0 and step % self._interval == 0

    def _should_run_iter(self) -> bool:
        return not self._by_epoch and self.__should_run(self.runner.iter_)

    def _should_run_epoch(self) -> bool:
        return (
            self._by_epoch
            and self.__should_run(self.epoch_based_trainer.epoch)
        )
