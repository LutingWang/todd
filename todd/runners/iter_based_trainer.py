__all__ = [
    'IterBasedTrainer',
]

import itertools
from typing import Any, Generator, TypeVar

from torch import nn

from ..patches.torch import set_epoch
from ..registries import RunnerRegistry
from .memo import Memo
from .trainer import Trainer

T = TypeVar('T', bound=nn.Module)


@RunnerRegistry.register_()
class IterBasedTrainer(Trainer[T]):

    def __init__(self, *args, iters: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # python interprets numbers like 2e3 as floats
        self._iters = int(iters)

    @property
    def iters(self) -> int:
        return self._iters

    def _iterate_dataloader(self) -> Generator[Any, None, None]:
        if self.inner_iter > 0:
            set_epoch(self._dataloader, self.epoch)
            yield from itertools.islice(
                self._dataloader,
                self.inner_iter,
                self.iters - self.iters_per_epoch * self.epoch,
            )
        while self._iter < self.iters:
            assert self.inner_iter == 0
            set_epoch(self._dataloader, self.epoch)
            yield from itertools.islice(
                self._dataloader,
                self.iters - self._iter,
            )

    def _setup(self) -> Memo:
        memo = super()._setup()
        memo['dataloader'] = self._iterate_dataloader()
        return memo
