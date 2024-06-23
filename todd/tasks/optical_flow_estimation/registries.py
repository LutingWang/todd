__all__ = [
    'OFEDatasetRegistry',
    'OpticalFlowRegistry',
]

from todd.registries import DatasetRegistry

from ..registries import OFERegistry


class OFEDatasetRegistry(OFERegistry, DatasetRegistry):
    pass


class OpticalFlowRegistry(OFERegistry):
    pass
