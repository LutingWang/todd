__all__ = [
    'BaseDataset',
]

import reprlib
from abc import ABC, abstractmethod
from typing import Generator, Generic, Iterator, Protocol, TypeVar

import torchvision.transforms as tf
from torch.utils.data import Dataset

from ..bases.configs import Config
from ..bases.registries import BuildPreHookMixin, Item, RegistryMeta
from ..loggers import logger
from ..patches.torch import get_world_size
from ..registries import TransformRegistry
from ..utils import Store
from .access_layers import BaseAccessLayer
from .registries import AccessLayerRegistry

KT_co = TypeVar('KT_co', covariant=True)
VT = TypeVar('VT')
T = TypeVar('T')


class KeysProtocol(Protocol[KT_co]):

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> KT_co:
        ...

    def __iter__(self) -> Iterator[KT_co]:
        ...


class BaseDataset(BuildPreHookMixin, Dataset[T], Generic[T, KT_co, VT], ABC):

    def __init__(
        self,
        *args,
        access_layer: BaseAccessLayer[KT_co, VT],
        transforms: tf.Compose | None = None,  # TODO: transforms in COCO
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._access_layer = access_layer
        self._transforms = transforms

        logger.debug("Initializing keys.")
        self._keys = self.build_keys()
        logger.debug(
            "Keys %s initialized with length %d",
            reprlib.repr(self._keys),
            len(self),
        )

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        if (access_layer := config.get('access_layer')) is not None:
            config.access_layer = AccessLayerRegistry.build_or_return(
                access_layer,
            )
        if (transforms := config.get('transforms')) is not None:
            config.transforms = TransformRegistry.build(
                Config(type=tf.Compose.__name__, transforms=transforms),
            )
        return config

    @property
    def access_layer(self) -> BaseAccessLayer[KT_co, VT]:
        return self._access_layer

    @property
    def transforms(self) -> tf.Compose | None:
        return self._transforms

    def build_keys(self) -> KeysProtocol[KT_co]:
        return list(self._access_layer)

    def __len__(self) -> int:
        if Store.DRY_RUN:
            return 4 * get_world_size()
        return len(self._keys)

    def __iter__(self) -> Generator[T, None, None]:
        for index in range(len(self)):
            yield self[index]

    def _access(self, index: int) -> tuple[KT_co, VT]:
        key = self._keys[index]
        return key, self._access_layer[key]

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        pass
