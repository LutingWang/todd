__all__ = [
    'Hook',
]

from ..registries import HookRegistry
from .base import BaseHook


@HookRegistry.register_()
class Hook(BaseHook):

    def _reset(self) -> None:
        self._tensor_ = None

    def _tensor(self):
        return self._tensor_

    def _register_tensor(self, tensor) -> None:
        self._tensor_ = tensor
