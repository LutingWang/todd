__all__ = [
    'ChainedScheduler',
]

import math
from typing import Iterable

from .base import SCHEDULERS, BaseScheduler, SchedulerConfig


@SCHEDULERS.register_module()
class ChainedScheduler(BaseScheduler):

    def __init__(self, schedulers: Iterable[SchedulerConfig]) -> None:
        self._schedulers = tuple(map(SCHEDULERS.build, schedulers))

    @property
    def value(self) -> float:
        return math.prod(self._schedulers)
