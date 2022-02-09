from .base import BaseScheduler
from .builder import SCHEDULERS


@SCHEDULERS.register_module()
class ConstantScheduler(BaseScheduler):
    @property
    def value(self) -> float:
        return self._value
