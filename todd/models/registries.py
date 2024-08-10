__all__ = [
    'NormRegistry',
    'ShadowRegistry',
    'FilterRegistry',
    'LossRegistry',
    'TorchVisionRegistry',
]

from ..registries import ModelRegistry


class NormRegistry(ModelRegistry):
    pass


class ShadowRegistry(ModelRegistry):
    pass


class FilterRegistry(ModelRegistry):
    pass


class LossRegistry(ModelRegistry):
    pass


class TorchVisionRegistry(ModelRegistry):
    pass
