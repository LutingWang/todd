__all__ = [
    'MultiCallsHook',
]

import torch

from .base import BaseHook, HookRegistry


@HookRegistry.register()
class MultiCallsHook(BaseHook):

    def _reset(self):
        self._tensors = []

    def _tensor(self) -> list[torch.Tensor]:
        return self._tensors

    def _register_tensor(self, tensor: torch.Tensor) -> None:
        self._tensors.append(tensor)
