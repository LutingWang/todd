__all__ = [
    'ODKDLossRegistry',
]

from todd_tasks import knowledge_distillation as kd

from ..registries import ODKDModelRegistry

KDLossRegistry = kd.models.KDLossRegistry


class ODKDLossRegistry(ODKDModelRegistry, KDLossRegistry):
    pass
