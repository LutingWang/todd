from typing import Any, Dict

from mmcv.utils import Registry

from ..adapts import AdaptLayer, AdaptModuleList


VISUALS = Registry('visuals')


class VisualLayer(AdaptLayer):
    def __init__(self, *args, registry: Registry = VISUALS, **kwargs):
        super().__init__(*args, registry=registry, **kwargs)

    def forward(self, hooked_tensors: Dict[str, Any], **kwargs):
        tensors = self._get_tensors(hooked_tensors)
        if self._multilevel:
            for i, level_tensors in enumerate(zip(*tensors)):
                self._adapt(*level_tensors, **kwargs, level=i) 
        else:
            self._adapt(*tensors, **kwargs)


class VisualModuleList(AdaptModuleList):
    def forward(self, hooked_tensors: Dict[str, Any]):
        for visual in self:
            visual(hooked_tensors)
