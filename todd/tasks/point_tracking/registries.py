__all__ = [
    'PTDatasetRegistry',
]

from todd.registries import DatasetRegistry

from ..registries import PTRegistry


class PTDatasetRegistry(PTRegistry, DatasetRegistry):
    pass
