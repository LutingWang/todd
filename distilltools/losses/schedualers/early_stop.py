from ...utils import get_iter

from .base import BaseSchedualer
from .builder import SCHEDUALERS


@SCHEDUALERS.register_module()
class EarlyStopSchedualer(BaseSchedualer):
    def __init__(self, *args, iter_: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._iter = iter_

    @property
    def value(self) -> float:
        return 0 if 0 <= self._iter <= get_iter() else self._value
