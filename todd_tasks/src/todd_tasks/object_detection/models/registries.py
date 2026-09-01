__all__ = [
    'ODLossRegistry',
]

from todd_torch.models import LossRegistry

from ..registries import ODModelRegistry


class ODLossRegistry(ODModelRegistry, LossRegistry):
    pass
