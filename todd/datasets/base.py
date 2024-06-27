__all__ = [
    'BaseDataset',
]

import reprlib
from abc import ABC, abstractmethod
from typing import Generator, Generic, TypeVar

from torch.utils.data import Dataset

from ..bases.registries import BuildSpec, BuildSpecMixin
from ..loggers import logger
from ..patches.py import classproperty
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

        logger.debug("Initializing keys.")
        self._keys = self.build_keys()
        logger.debug(
            "Keys %s initialized with length %d",
            reprlib.repr(self._keys),
            len(self),
        )

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(access_layer=AccessLayerRegistry.build)
        return super().build_spec | build_spec

    @property
    def access_layer(self) -> BaseAccessLayer[KT, VT]:
        return self._access_layer

    def build_keys(self) -> list[KT]:
        return list(self._access_layer)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Generator[T, None, None]:
        for index in range(len(self)):
            yield self[index]

    def _access(self, index: int) -> tuple[KT, VT]:
        key = self._keys[index]
        return key, self._access_layer[key]

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        pass
