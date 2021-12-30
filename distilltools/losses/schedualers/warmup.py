from ...utils import get_iter

from .base import BaseSchedualer
from .builder import SCHEDUALERS


@SCHEDUALERS.register_module()
class WarmupSchedualer(BaseSchedualer):
    def __init__(self, *args, iter_: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._iter = iter_

    @property
    def value(self) -> float:
        weight = min(get_iter() / self._iter, 1) if self._iter > 0 else 1
        return weight * self._value
