from collections.abc import Iterable
from typing import Any, Dict, List

import einops.layers.torch as einops
from mmcv.runner import BaseModule, ModuleDict
from mmcv.utils import Registry
import torch.nn as nn


ADAPTS= Registry('adapts')
ADAPTS.register_module(module=nn.Conv2d)
ADAPTS.register_module(module=nn.Linear)
ADAPTS.register_module(module=einops.Rearrange)
ADAPTS.register_module(module=einops.Reduce)


class AdaptLayer(BaseModule):
    def __init__(self, cfg: dict, registry: Registry = ADAPTS, **kwargs):
        super().__init__(**kwargs)
        self._tensor_names: List[str] = cfg.pop('tensor_names')
        self._multilevel: bool = cfg.pop('multilevel', False)
        self._adapt = registry.build(cfg)

    def forward(self, hooked_tensors: Dict[str, Any], **kwargs):
        tensors = [
            hooked_tensors[tensor_name] 
            for tensor_name in self._tensor_names
        ]
        if self._multilevel:
            return [
                self._adapt(*level_tensors, **kwargs) 
                for level_tensors in zip(*tensors)
            ]
        else:
            return self._adapt(*tensors, **kwargs)


class AdaptModuleDict(ModuleDict):
    def __init__(self, adapts: dict, **kwargs):
        adapts = {
            k: AdaptLayer.build(v)
            for k, v in adapts.items()
        }
        super().__init__(modules=adapts, **kwargs)

    def forward(self, hooked_tensors: Dict[str, Any], inplace: bool = False) -> Dict[str, Any]:
        if not inplace:
            hooked_tensors = dict(hooked_tensors)
        adapted_tensors = dict()
        for name, adapt in self.items():
            tensors = adapt(hooked_tensors)
            if isinstance(name, str):
                update_tensors = {name: tensors}
            elif isinstance(name, Iterable):
                assert len(name) == len(adapted_tensors)
                update_tensors = dict(zip(name, adapted_tensors))
            else:
                raise NotImplementedError
            hooked_tensors.update(update_tensors)
            adapted_tensors.update(update_tensors)
        return adapted_tensors
