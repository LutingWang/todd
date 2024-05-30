__all__ = [
    'BaseDataset',
]

import reprlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch.utils.data import Dataset

from ..loggers import logger
from ..patches.py import classproperty
from ..registries import BuildSpec, BuildSpecMixin
from .access_layers import BaseAccessLayer
from .registries import AccessLayerRegistry

KT = TypeVar('KT')
VT = TypeVar('VT')
T = TypeVar('T')


class BaseDataset(BuildSpecMixin, Dataset[T], Generic[T, KT, VT], ABC):

    def __init__(
        self,
        *args,
        access_layer: BaseAccessLayer[KT, VT],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._access_layer = access_layer
        self._build_keys()

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(access_layer=AccessLayerRegistry.build)
        return super().build_spec | build_spec

    @property
    def access_layer(self) -> BaseAccessLayer[KT, VT]:
        return self._access_layer

    def _build_keys(self) -> None:
        logger.debug("Initializing keys.")
        self._keys = list(self._access_layer)
        logger.debug(
            "Keys %s initialized with length %d",
            reprlib.repr(self._keys),
            len(self),
        )

    def __len__(self) -> int:
        return len(self._keys)

    def _access(self, index: int) -> VT:
        key = self._keys[index]
        return self._access_layer[key]

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        pass
