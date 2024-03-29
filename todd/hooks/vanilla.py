__all__ = [
    'VanillaHook',
]

from ..base import HookRegistry
from .base import BaseHook


@HookRegistry.register_()
class VanillaHook(BaseHook):

    def _reset(self) -> None:
        self._tensor_ = None

    def _tensor(self):
        return self._tensor_

    def _register_tensor(self, tensor) -> None:
        self._tensor_ = tensor
