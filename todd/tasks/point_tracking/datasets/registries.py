__all__ = [
    'PTAccessLayerRegistry',
]

from todd.datasets.registries import AccessLayerRegistry

from ...registries import PTRegistry


class PTAccessLayerRegistry(PTRegistry, AccessLayerRegistry):
    pass
