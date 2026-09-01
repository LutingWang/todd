__all__ = [
    'Store',
]

from todd.patches import classproperty
from todd.utils import StoreMeta

from ..patches.torch import get_device


class Store(metaclass=StoreMeta):
    DEVICE: str = get_device()
    TRAIN_WITH_VAL_DATASET: bool

    @classmethod
    def _device(cls, name: str) -> bool:
        return cls.DEVICE == name

    @classproperty
    def cpu(self) -> bool:
        return self._device('cpu')

    @classproperty
    def cuda(self) -> bool:
        return self._device('cuda')

    @classproperty
    def mps(self) -> bool:
        return self._device('mps')
