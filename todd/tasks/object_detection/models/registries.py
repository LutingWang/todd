__all__ = [
    'ODLossRegistry',
]

from todd.models import LossRegistry

from ..registries import ODModelRegistry


class ODLossRegistry(ODModelRegistry, LossRegistry):
    pass
