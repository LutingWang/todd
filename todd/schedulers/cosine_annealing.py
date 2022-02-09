import math
from typing import Optional
from ..utils import get_iter

from .base import BaseScheduler
from .builder import SCHEDULERS


@SCHEDULERS.register_module()
class CosineAnnealingScheduler(BaseScheduler):
    def __init__(self, *args, min_value: float = 0, T: int, T_mult: int = 1, T_max: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_iter = 0
        self._min_value = min_value
        self._T = T
        self._T_mult = T_mult
        self._T_max = T_max

    @property
    def value(self) -> float:
        T_cur = get_iter() - self._last_iter
        while T_cur >= self._T:
            T_cur -= self._T
            self._last_iter += self._T
            self._T *= self._T_mult
            if self._T_max is not None and self._T > self._T_max:
                self._T = self._T_max
        weight = (1 + math.cos(math.pi * T_cur / self._T)) / 2
        return weight * (self._value - self._min_value) + self._min_value
