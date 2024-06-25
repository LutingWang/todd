__all__ = [
    'ODKDLossRegistry',
]

import todd.tasks.knowledge_distillation as kd

from ..registries import ODKDModelRegistry

KDLossRegistry = kd.models.KDLossRegistry


class ODKDLossRegistry(ODKDModelRegistry, KDLossRegistry):
    pass
