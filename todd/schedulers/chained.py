__all__ = [
    'ChainedScheduler',
]

import math
from typing import Iterable

from .base import SCHEDULERS, BaseScheduler, SchedulerConfig


@SCHEDULERS.register_module()
class ChainedScheduler(BaseScheduler):

    def __init__(
        self,
        schedulers: Iterable[SchedulerConfig],
        value: float = 1.0,
    ) -> None:
        self._schedulers = tuple(map(SCHEDULERS.build, schedulers))
        self._value = value

    @property
    def value(self) -> float:
        return math.prod(self._schedulers) * self._value
