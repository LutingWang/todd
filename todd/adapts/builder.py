from typing import Any, Dict, Iterable, List, Union

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
    REGISTRY = ADAPTS

    def __init__(self, cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self._id = cfg.pop('id_')
        self._tensor_names: List[str] = cfg.pop('tensor_names')
        self._multilevel: bool = cfg.pop('multilevel', False)
        self._adapt = self.REGISTRY.build(cfg)

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


AdaptLayerCfg = Union[dict, AdaptLayer]


class AdaptModuleList(ModuleList):
    LAYER_TYPE = AdaptLayer

    def __init__(
        self, 
        adapts: Union[Dict[str, AdaptLayerCfg], List[AdaptLayerCfg]], 
        **kwargs,
    ):
        if isinstance(adapts, dict):
            adapts = [
                dict(id_=id_, **adapt) if isinstance(adapt, dict) else adapt 
                for id_, adapt in adapts.items()
            ]
        adapts = [
            self.LAYER_TYPE.build(adapt) for adapt in adapts
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


AdaptModuleListCfg = Union[Dict[str, AdaptLayerCfg], List[AdaptLayerCfg], AdaptModuleList]
