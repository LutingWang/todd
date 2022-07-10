import math
import numbers
from abc import abstractmethod
from typing import Any, Optional, Union

from ..base import Registry, get_iter

__all__ = [
    'BaseScheduler',
    'SCHEDULERS',
    'SchedulerConfig',
    'IntervalScheduler',
]


class BaseScheduler(numbers.Real):

    @property
    @abstractmethod
    def value(self) -> float:
        pass

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __neg__(self):
        return -self.value

    def __pos__(self):
        return +self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __pow__(self, exponent):
        return self.value**exponent

    def __rpow__(self, base):
        return base**self.value

    def __abs__(self):
        return abs(self.value)

    def __eq__(self, other):
        return self.value == other

    def __float__(self) -> float:
        return float(self.value)

    def __trunc__(self) -> int:
        return math.trunc(self.value)

    def __floor__(self) -> int:
        return math.floor(self.value)

    def __ceil__(self) -> int:
        return math.ceil(self.value)

    def __round__(self, ndigits=None):
        return round(self.value, ndigits)

    def __floordiv__(self, other):
        return self.value // other

    def __rfloordiv__(self, other):
        return other // self.value

    def __mod__(self, other):
        return self.value % other

    def __rmod__(self, other):
        return other % self.value

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other


SCHEDULERS: Registry[Any] = Registry(
    'schedulers',
    base=BaseScheduler,
)
SchedulerConfig = Union[dict, BaseScheduler]


class IntervalScheduler(BaseScheduler):

    def __init__(
        self,
        *,
        start_value: float,
        end_value: float,
        start_iter: int = 0,
        end_iter: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._start_value = start_value
        self._end_value = end_value
        self._start_iter = start_iter
        self._end_iter = float('inf') if end_iter is None else end_iter

    @abstractmethod
    def _weight(
        self,
        cur_iter: int,
        total_iter: float,  # may be float('inf')
    ) -> float:
        pass

    @property
    def value(self) -> float:
        if get_iter() <= self._start_iter:
            return self._start_value
        if get_iter() >= self._end_iter:
            return self._end_value
        weight = self._weight(
            get_iter() - self._start_iter,
            self._end_iter - self._start_iter,
        )
        return (
            weight * (self._end_value - self._start_value) + self._start_value
        )
