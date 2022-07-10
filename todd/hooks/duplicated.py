from typing import Any, List

from .base import HOOKS
from .standard import StandardHook


@HOOKS.register_module()
class DuplicatedHook(StandardHook):

    def __init__(self, *args, num: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._num = num

    def _tensor(self) -> List[Any]:
        return [super()._tensor()] * self._num
