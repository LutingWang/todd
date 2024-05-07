__all__ = [
    'BaseAccessLayer',
    'BaseDataset',
]

import reprlib
from abc import ABC, abstractmethod
from typing import Generic, MutableMapping, TypeVar

from torch.utils.data import Dataset

from ..configs import Config
from ..logger import logger
from .registries import AccessLayerRegistry

KT = TypeVar('KT')
VT = TypeVar('VT')


class BaseAccessLayer(MutableMapping[KT, VT]):

    def __init__(
        self,
        data_root: str,
        task_name: str = '',
    ) -> None:
        self._data_root = data_root
        self._task_name = task_name

    @property
    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def touch(self) -> None:
        pass


T = TypeVar('T')


class BaseDataset(Dataset[T], Generic[T, KT, VT], ABC):

    def __init__(
        self,
        *args,
        access_layer: Config,
        keys: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._build_access_layer(access_layer)
        if keys is None:
            keys = Config()
        self._build_keys(keys)

    @property
    def access_layer(self) -> BaseAccessLayer[KT, VT]:
        return self._access_layer

    def _build_access_layer(self, config: Config) -> None:
        self._access_layer: BaseAccessLayer[KT, VT] = \
            AccessLayerRegistry.build(config)

    def _build_keys(self, config: Config) -> None:
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
