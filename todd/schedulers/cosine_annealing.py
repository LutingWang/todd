import math
from typing import Optional

from .base import SCHEDULERS, BaseScheduler

__all__ = [
    'CosineAnnealingScheduler',
]


@SCHEDULERS.register_module()
class CosineAnnealingScheduler(BaseScheduler):

    def __init__(
        self,
        *args,
        T: int,
        T_mult: int = 1,
        T_max: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._last_iter = 0
        self._T = T
        self._T_mult = T_mult
        self._T_max = T_max

    def _weight(self, cur_iter: int, _) -> float:
        T_cur = cur_iter - self._last_iter
        while T_cur >= self._T:
            T_cur -= self._T
            self._last_iter += self._T
            self._T *= self._T_mult
            if self._T_max is not None and self._T > self._T_max:
                self._T = self._T_max
        return -(1 + math.cos(math.pi * T_cur / self._T)) / 2
