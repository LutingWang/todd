__all__ = [
    'VanillaHook',
]

from ..registries import HookRegistry
from .base import BaseHook


@HookRegistry.register_()
class VanillaHook(BaseHook):  # TODO: Rename to hook

    def _reset(self) -> None:
        self._tensor_ = None

    def _tensor(self):
        return self._tensor_

    def _register_tensor(self, tensor) -> None:
        self._tensor_ = tensor
