from .base import BaseSchedualer
from .builder import SCHEDUALERS


@SCHEDUALERS.register_module()
class ConstantSchedualer(BaseSchedualer):
    @property
    def value(self) -> float:
        return self._value
