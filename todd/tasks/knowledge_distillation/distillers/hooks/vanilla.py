__all__ = [
    'Hook',
]

from ..registries import KDHookRegistry
from .base import BaseHook


@KDHookRegistry.register_()
class Hook(BaseHook):

    def _reset(self) -> None:
        self._tensor_ = None

    def _tensor(self):
        return self._tensor_

    def _register_tensor(self, tensor) -> None:
        self._tensor_ = tensor
