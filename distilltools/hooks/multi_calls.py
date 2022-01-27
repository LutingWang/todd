from typing import Dict, List

import torch

from .base import BaseHook
from .builder import HOOKS


@HOOKS.register_module()
class MultiCallsHook(BaseHook):
    @property
    def tensor(self) -> Dict[str, List[torch.Tensor]]:
        return {self.id_: self._tensors}

    def reset(self):
        self._tensors = []

    def register_tensor(self, tensor: torch.Tensor):
        if self._detach:
            tensor = tensor.detach()
        self._tensors.append(tensor)
