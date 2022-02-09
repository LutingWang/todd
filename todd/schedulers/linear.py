from ..utils import get_iter

from .base import BaseScheduler
from .builder import SCHEDULERS


@SCHEDULERS.register_module()
class LinearScheduler(BaseScheduler):
    def __init__(self, *args, end_value: float, iter_: int, end_iter: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._end_value = end_value
        self._iter = iter_
        self._end_iter = end_iter

    @property
    def value(self) -> float:
        if get_iter() <= self._iter:
            return self._value
        if get_iter() >= self._end_iter:
            return self._end_value
        weight = (get_iter() - self._iter) / (self._end_iter - self._iter)
        return weight * (self._end_value - self._value) + self._value


@SCHEDULERS.register_module()
class WarmupScheduler(LinearScheduler):
    def __init__(self, *args, value: int = 1, iter_: int, **kwargs):
        super().__init__(
            *args, value=0, end_value=value,
            iter_=0, end_iter=iter_, **kwargs,
        )

@SCHEDULERS.register_module()
class EarlyStopScheduler(LinearScheduler):
    def __init__(self, *args, iter_: int, **kwargs):
        super().__init__(
            *args, end_value=0, iter_=iter_, end_iter=iter_, **kwargs,
        )
