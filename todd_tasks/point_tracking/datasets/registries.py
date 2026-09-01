__all__ = [
    'PTAccessLayerRegistry',
]

from todd_tasks.registries import PTRegistry
from todd_torch.datasets import AccessLayerRegistry


class PTAccessLayerRegistry(PTRegistry, AccessLayerRegistry):
    pass
