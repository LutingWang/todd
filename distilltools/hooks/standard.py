from typing import Dict

import torch

from .base import BaseHook
from .builder import HOOKS


@HOOKS.register_module()
class StandardHook(BaseHook):
    @property
    def tensor(self) -> Dict[str, torch.Tensor]:
        return {self.alias: self._tensor}

    def reset(self):
        self._tensor = None

    def register_tensor(self, tensor: torch.Tensor):
        if self._detach:
            tensor = tensor.detach()
        self._tensor = tensor
