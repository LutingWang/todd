from typing import List, cast

from ..base import get_iter
from .base import SCHEDULERS, BaseScheduler

__all__ = [
    'StepScheduler',
]


@SCHEDULERS.register_module()
class StepScheduler(BaseScheduler):

    def __init__(
        self,
        *args,
        value: float = 1.0,
        iters: List[int],
        ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        values = [value * ratio**i for i in range(len(iters))]
        self._values = [0.0] + values
        self._iters = cast(List[float], iters) + [float('inf')]

    @property
    def value(self) -> float:
        iter_ = get_iter()
        for value, iter in zip(self._values, self._iters):
            if iter > iter_:
                return value
        raise RuntimeError(iter_, self._values, self._iters)
