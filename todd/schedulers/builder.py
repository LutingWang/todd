from mmcv.utils import Registry

from ..adapts import AdaptLayer, AdaptModuleList


SCHEDULERS = Registry('schedulers')


class SchedulerLayer(AdaptLayer):
    REGISTRY = SCHEDULERS

    def __init__(self, cfg: dict, **kwargs):
        cfg['id_'] = cfg['tensor_names']
        super().__init__(cfg, **kwargs)


class SchedulerModuleList(AdaptModuleList):
    LAYER_TYPE = SchedulerLayer
