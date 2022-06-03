import itertools
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import einops.layers.torch as einops
from mmcv.cnn import CONV_LAYERS, PLUGIN_LAYERS
from mmcv.runner import BaseModule, ModuleList
from mmcv.utils import Registry
import torch.nn as nn


ADAPTS= Registry('adapts')
ADAPTS.module_dict.update(PLUGIN_LAYERS.module_dict)
ADAPTS.module_dict.update(CONV_LAYERS.module_dict)
ADAPTS.register_module(module=nn.Linear)
ADAPTS.register_module(module=einops.Rearrange)
ADAPTS.register_module(module=einops.Reduce)


class AdaptLayer(BaseModule):
    REGISTRY = ADAPTS

    def __init__(self, cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self._id = cfg.pop('id_')
        self._tensor_names: List[str] = cfg.pop('tensor_names')
        self._multilevel: Union[bool, int] = cfg.pop('multilevel', False)
        if isinstance(self._multilevel, bool):
            self._adapt = self.REGISTRY.build(cfg)
        elif isinstance(self._multilevel, int):
            self._adapt = ModuleList([
                self.REGISTRY.build(cfg) for _ in range(self._multilevel)
            ])
        else:
            assert False

    @property
    def name(self) -> str:
        if isinstance(self._id, str):
            return self._id
        return str(self._id)

    def _get_tensors(
        self, 
        tensors: Dict[str, Any], 
        tensor_names: Optional[List[str]] = None,
    ) -> list:
        tensor_names = self._tensor_names if tensor_names is None else tensor_names
        return [
            tensors[tensor_name] for tensor_name in tensor_names
        ]

    @property
    def adapts(self) -> Iterator[nn.Module]:
        assert self._multilevel
        adapts = (
            self._adapt if isinstance(self._adapt, ModuleList) else 
            itertools.repeat(self._adapt)
        )
        return adapts

    def _adapt_tensors(self, tensors: list, kwargs: dict):
        if not self._multilevel:
            return self._adapt(*tensors, **kwargs)
        return [
            adapt(*level_tensors, **kwargs) 
            for adapt, *level_tensors in zip(self.adapts, *tensors)
        ]

    def forward(self, hooked_tensors: Dict[str, Any], **kwargs) -> dict:
        tensors = self._get_tensors(hooked_tensors)
        adapted_tensors = self._adapt_tensors(tensors, kwargs)

        if isinstance(self._id, str):
            return {self._id: adapted_tensors}
        if isinstance(self._id, Sequence):
            if self._multilevel:
                assert all(len(self._id) == len(level_tensors) for level_tensors in adapted_tensors)
                adapted_tensors = zip(*adapted_tensors)
            assert len(self._id) == len(adapted_tensors), self._id
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
