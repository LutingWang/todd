from typing import Any, Dict

from mmcv.utils import Registry

from ..adapts import AdaptLayer, AdaptModuleList

VISUALS = Registry('visuals')


class VisualLayer(AdaptLayer):
    REGISTRY = VISUALS

    def _adapt_tensors(self, tensors: list, kwargs: dict):
        if not self._multilevel:
            return self._adapt(*tensors, **kwargs)

        from .savers import BaseSaver

        adapted_tensors = []
        for level, (adapt, *level_tensors) in \
            enumerate(zip(self.adapts, *tensors)):
            level_kwargs = kwargs
            if isinstance(adapt, BaseSaver):
                level_kwargs = dict(level=level, **level_kwargs)
            adapted_tensors.append(adapt(*level_tensors, **level_kwargs))
        return adapted_tensors


class VisualModuleList(AdaptModuleList):
    LAYER_TYPE = VisualLayer
