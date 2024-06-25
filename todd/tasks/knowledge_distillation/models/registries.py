__all__ = [
    'KDLossRegistry',
]

from todd.models import LossRegistry

from ..registries import KDModelRegistry


class KDLossRegistry(KDModelRegistry, LossRegistry):
    pass
