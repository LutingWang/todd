__all__ = [
    'IterBasedTrainer',
]

import itertools
from typing import Any

from ..base import RunnerRegistry
from .trainer import Trainer

Memo = dict[str, Any]


@RunnerRegistry.register_()
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
