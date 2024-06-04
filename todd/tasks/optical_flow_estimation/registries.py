__all__ = [
    'OFEDatasetRegistry',
]

from ...registries import DatasetRegistry
from ..registries import OFERegistry


class OFEDatasetRegistry(OFERegistry, DatasetRegistry):
    pass
