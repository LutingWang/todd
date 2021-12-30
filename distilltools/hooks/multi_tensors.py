from typing import Dict, List

import torch

from .base import BaseHook
from .builder import HOOKS


@HOOKS.register_module()
class MultiTensorsHook(BaseHook):
    def __init__(self, tensor_names: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_names = [
            f'{self.alias}_{tensor_name}' 
            for tensor_name in tensor_names
        ]

    @property
    def tensor(self) -> Dict[str, torch.Tensor]:
        return dict(zip(self.tensor_names, self._tensors))

    def reset(self):
        self._tensors = []

    def register_tensor(self, tensors: List[torch.Tensor]):
        self._tensors = tensors
