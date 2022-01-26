from typing import Any, Dict, Iterable, List

import einops.layers.torch as einops
from mmcv.runner import BaseModule, ModuleList
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
        self._id = cfg.pop('id_')
        self._tensor_names: List[str] = cfg.pop('tensor_names')
        self._multilevel: bool = cfg.pop('multilevel', False)
        self._adapt = registry.build(cfg)

    @property
    def name(self) -> str:
        if isinstance(self._id, str):
            return self._id
        return str(self._id)

    def _get_tensors(self, hooked_tensors: Dict[str, Any]) -> Dict[str, Any]:
        tensors = [
            hooked_tensors[tensor_name] 
            for tensor_name in self._tensor_names
        ]
        return tensors

    def forward(self, hooked_tensors: Dict[str, Any], **kwargs) -> dict:
        tensors = self._get_tensors(hooked_tensors)
        if self._multilevel:
            adapted_tensors = [
                self._adapt(*level_tensors, **kwargs) 
                for level_tensors in zip(*tensors)
            ]
        else:
            adapted_tensors = self._adapt(*tensors, **kwargs)

        if isinstance(self._id, str):
            return {self._id: adapted_tensors}
        if isinstance(self._id, Iterable):
            assert len(self._id) == len(adapted_tensors)
            return dict(zip(self._id, adapted_tensors))
        raise NotImplementedError


class AdaptModuleList(ModuleList):
    def __init__(self, adapts: List[dict], **kwargs):
        adapts = [
            AdaptLayer.build(adapt) for adapt in adapts
        ]
        super().__init__(modules=adapts, **kwargs)

    def forward(self, hooked_tensors: Dict[str, Any], inplace: bool = False) -> Dict[str, Any]:
        if not inplace:
            hooked_tensors = dict(hooked_tensors)
        adapted_tensors = dict()
        for adapt in self:
            tensors = adapt(hooked_tensors)
            hooked_tensors.update(tensors)
            adapted_tensors.update(tensors)
        return adapted_tensors
