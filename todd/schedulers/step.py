from typing import cast

from ..base import Store
from .base import BaseScheduler, SchedulerRegistry

__all__ = [
    'StepScheduler',
]


@SchedulerRegistry.register()
class StepScheduler(BaseScheduler):

    def __init__(
        self,
        *args,
        value: float = 1.0,
        iters: list[int],
        ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        values = [value * ratio**i for i in range(len(iters))]
        self._values = [0.0] + values
        self._iters = cast(list[float], iters) + [float('inf')]

    @property
    def value(self) -> float:
        for value, iter_ in zip(self._values, self._iters):
            if iter_ > Store.ITER:
                return value
        raise RuntimeError(Store.ITER, self._values, self._iters)
