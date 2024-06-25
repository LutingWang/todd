__all__ = [
    'AppendHook',
]

import torch

from ..registries import KDHookRegistry
from .base import BaseHook


@KDHookRegistry.register_()
class AppendHook(BaseHook):

    def _reset(self) -> None:
        self._tensors: list[torch.Tensor] = []

    def _tensor(self) -> list[torch.Tensor]:
        return self._tensors

    def _register_tensor(self, tensor: torch.Tensor) -> None:
        self._tensors.append(tensor)
