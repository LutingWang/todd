from typing import Dict, List

import torch

from .base import BaseHook
from .builder import HOOKS


@HOOKS.register_module()
class MultiTensorsHook(BaseHook):

    def __init__(self, tensor_names: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._id_tensor_names = [
            f'{self.id_}_{tensor_name}' for tensor_name in tensor_names
        ]

    @property
    def tensor(self) -> Dict[str, torch.Tensor]:
        return dict(zip(self._id_tensor_names, self._tensors))

    def reset(self):
        self._tensors = []

    def register_tensor(self, tensors: List[torch.Tensor]):
        if self._detach:
            tensors = [tensor.detach() for tensor in tensors]
        self._tensors = tensors
