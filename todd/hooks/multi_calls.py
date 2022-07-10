from typing import List

import torch

from .base import HOOKS, BaseHook


@HOOKS.register_module()
class MultiCallsHook(BaseHook):

    def _reset(self):
        self._tensors = []

    def _tensor(self) -> List[torch.Tensor]:
        return self._tensors

    def _register_tensor(self, tensor: torch.Tensor) -> None:
        self._tensors.append(tensor)
