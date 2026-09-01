__all__ = [
    'KDLossRegistry',
]

from todd_torch.models import LossRegistry

from ..registries import KDModelRegistry


class KDLossRegistry(KDModelRegistry, LossRegistry):
    pass
