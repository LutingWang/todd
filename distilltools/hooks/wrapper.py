from typing import Any, Dict, List

from mmcv.runner import BaseModule
from torch import nn


class HookWrapper(BaseModule):
    def __init__(
        self, module: nn.Module, tensor_names: List[str], multilevel: bool = False, **kwargs,
    ):
        super().__init__(**kwargs)

        self._module = module
        self._tensor_names = tensor_names
        self._multilevel = multilevel

    def forward(self, hooked_tensors: Dict[str, Any], **kwargs):
        tensors = [
            hooked_tensors[tensor_name] 
            for tensor_name in self._tensor_names
        ]
        if self._multilevel:
            return [
                self._module(*level_tensors, **kwargs) 
                for level_tensors in zip(*tensors)
            ]
        else:
            return self._module(*tensors, **kwargs)
