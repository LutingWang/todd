__all__ = [
    'PTAccessLayerRegistry',
]

from todd_torch.datasets import AccessLayerRegistry

from ...registries import PTRegistry


class PTAccessLayerRegistry(PTRegistry, AccessLayerRegistry):
    pass
