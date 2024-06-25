__all__ = [
    'OFEAccessLayerRegistry',
]

from todd.datasets.registries import AccessLayerRegistry

from ..registries import OFEDatasetRegistry


class OFEAccessLayerRegistry(OFEDatasetRegistry, AccessLayerRegistry):
    pass
