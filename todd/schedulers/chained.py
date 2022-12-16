__all__ = [
    'ChainedScheduler',
]

import math
from typing import Iterable

from ..base import Config
from .base import BaseScheduler, SchedulerRegistry


@SchedulerRegistry.register()
class ChainedScheduler(BaseScheduler):

    def __init__(
        self,
        schedulers: Iterable[Config],
        value: float = 1.0,
    ) -> None:
        self._schedulers = tuple(map(SchedulerRegistry.build, schedulers))
        self._value = value

    @property
    def value(self) -> float:
        return math.prod(self._schedulers) * self._value
