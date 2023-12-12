__all__ = [
    'ConcatAccessLayer',
]

from abc import ABC
from typing import Iterable, Iterator, TypeVar

from ..base import AccessLayerRegistry, Config
from .base import BaseAccessLayer

KT = TypeVar('KT')
VT = TypeVar('VT')


@AccessLayerRegistry.register_()
class ConcatAccessLayer(BaseAccessLayer[KT, VT], ABC):

    def __init__(
        self,
        *args,
        access_layers: Iterable[Config],
        **kwargs,
    ) -> None:
        self._access_layers: list[BaseAccessLayer[KT, VT]] = list(
            map(AccessLayerRegistry.build, access_layers)
        )
        data_root = '|'.join(
            access_layer._data_root for access_layer in self._access_layers
        )
        super().__init__(data_root, *args, **kwargs)

    def _access_layer_with_key(self, key: KT) -> BaseAccessLayer[KT, VT]:
        access_layers = [
            access_layer for access_layer in self._access_layers
            if key in access_layer
        ]
        assert len(access_layers) == 1
        return access_layers[0]

    @property
    def exists(self) -> bool:
        return all(access_layer.exists for access_layer in self._access_layers)

    def touch(self) -> None:
        for access_layer in self._access_layers:
            access_layer.touch()

    def __iter__(self) -> Iterator[KT]:
        for access_layer in self._access_layers:
            yield from access_layer

    def __len__(self) -> int:
        return sum(map(len, self._access_layers))

    def __getitem__(self, key: KT) -> VT:
        access_layer = self._access_layer_with_key(key)
        return access_layer.__getitem__(key)

    def __setitem__(self, key: KT, value: VT) -> None:
        access_layer = self._access_layer_with_key(key)
        access_layer.__setitem__(key, value)

    def __delitem__(self, key: KT) -> None:
        access_layer = self._access_layer_with_key(key)
        access_layer.__delitem__(key)
