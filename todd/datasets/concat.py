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
        return sum(len(access_layer) for access_layer in self._access_layers)

    def __delitem__(self, key: KT) -> None:
        for access_layer in self._access_layers:
            if key in access_layer:
                del access_layer[key]
