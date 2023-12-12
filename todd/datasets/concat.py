__all__ = [
    'ConcatAccessLayer',
]

from abc import ABC
from typing import Generator, TypeVar

from ..base import AccessLayerRegistry, Config
from .base import BaseAccessLayer

VT = TypeVar('VT')


@AccessLayerRegistry.register_()
class ConcatAccessLayer(BaseAccessLayer[str, VT], ABC):
    KEY_SEPARATOR = ':'
    DATA_ROOT_SEPARATOR = '|'

    def __init__(
        self,
        *args,
        access_layers: Config,
        **kwargs,
    ) -> None:
        named_access_layers: dict[str, BaseAccessLayer[str, VT]] = {
            k: AccessLayerRegistry.build(v)
            for k, v in access_layers.items()
        }
        assert all(self.KEY_SEPARATOR not in k for k in named_access_layers)

        data_root = self.DATA_ROOT_SEPARATOR.join(
            access_layer._data_root
            for access_layer in named_access_layers.values()
        )
        super().__init__(data_root, *args, **kwargs)

        self._named_access_layers = named_access_layers

    def _parse(self, key: str) -> tuple[BaseAccessLayer[str, VT], str]:
        name, key = key.split(self.KEY_SEPARATOR, maxsplit=1)
        return self._named_access_layers[name], key

    @property
    def exists(self) -> bool:
        return all(
            access_layer.exists
            for access_layer in self._named_access_layers.values()
        )

    def touch(self) -> None:
        for access_layer in self._named_access_layers.values():
            access_layer.touch()

    def __iter__(self) -> Generator[str, None, None]:
        for name, access_layer in self._named_access_layers.items():
            for k in access_layer:
                yield name + self.KEY_SEPARATOR + k

    def __len__(self) -> int:
        return sum(map(len, self._named_access_layers))

    def __getitem__(self, key: str) -> VT:
        access_layer, key = self._parse(key)
        return access_layer.__getitem__(key)

    def __setitem__(self, key: str, value: VT) -> None:
        access_layer, key = self._parse(key)
        access_layer.__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        access_layer, key = self._parse(key)
        access_layer.__delitem__(key)
