from mmcv.utils import Registry

from ..base import ModuleJob, ModuleStep

__all__ = [
    'VISUALS',
    'VisualLayer',
    'VisualModuleList',
]

VISUALS = Registry('visuals')


class VisualLayer(ModuleStep):
    REGISTRY = VISUALS

    def _forward(self, inputs: tuple, kwargs: dict) -> tuple:
        raise NotImplementedError
        # if not self._parallel:
        #     return self._adapt(*tensors, **kwargs)

        # adapted_tensors = []
        # for level, (adapt, *level_tensors) in \
        #         enumerate(zip(self.adapts, *tensors)):
        #     level_kwargs = kwargs
        #     if isinstance(adapt, BaseSaver):
        #         level_kwargs = dict(level=level, **level_kwargs)
        #     adapted_tensors.append(adapt(*level_tensors, **level_kwargs))
        # return adapted_tensors


class VisualModuleList(ModuleJob):
    LAYER_TYPE = VisualLayer
