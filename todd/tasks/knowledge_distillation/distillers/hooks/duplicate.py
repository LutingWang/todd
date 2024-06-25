__all__ = [
    'DuplicateHook',
]

from ..registries import KDHookRegistry
from .vanilla import Hook


@KDHookRegistry.register_()
class DuplicateHook(Hook):

    def __init__(self, *args, num: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._num = num

    def _tensor(self) -> list:
        return [super()._tensor()] * self._num
