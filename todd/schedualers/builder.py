from mmcv.utils import Registry

from ..adapts import AdaptLayer, AdaptModuleList


SCHEDUALERS = Registry('schedualers')


class SchedualerLayer(AdaptLayer):
    REGISTRY = SCHEDUALERS

    def __init__(self, cfg: dict, **kwargs):
        cfg['id_'] = cfg['tensor_names']
        super().__init__(cfg, **kwargs)


class SchedualerModuleList(AdaptModuleList):
    LAYER_TYPE = SchedualerLayer
