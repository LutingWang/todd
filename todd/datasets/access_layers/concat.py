__all__ = [
    'ConcatAccessLayer',
]

from abc import ABC
from typing import Any, Generator, Mapping, TypeVar

from ...bases.configs import Config
from ...bases.registries import BuildPreHookMixin, Item, RegistryMeta
from ..registries import AccessLayerRegistry
from .base import BaseAccessLayer

VT = TypeVar('VT')


@AccessLayerRegistry.register_()
class ConcatAccessLayer(BuildPreHookMixin, BaseAccessLayer[str, VT], ABC):
    KEY_SEPARATOR = ':'
    DATA_ROOT_SEPARATOR = '|'

    def __init__(
        self,
        *args,
        access_layers: Mapping[str, BaseAccessLayer[str, VT]],
        **kwargs,
    ) -> None:
        assert not any(self.KEY_SEPARATOR in k for k in access_layers)

        data_root = self.DATA_ROOT_SEPARATOR.join(
            al._data_root for al in access_layers.values()
        )
        super().__init__(data_root, *args, **kwargs)

        self._access_layers = dict(access_layers)

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        access_layers: dict[str, Any] = config.access_layers
        config.access_layers = {
            k: AccessLayerRegistry.build_or_return(v)
            for k, v in access_layers.items()
        }
        return config

    def _parse(self, key: str) -> tuple[BaseAccessLayer[str, VT], str]:
        name, key = key.split(self.KEY_SEPARATOR, maxsplit=1)
        return self._access_layers[name], key

    @property
    def exists(self) -> bool:
        return all(al.exists for al in self._access_layers.values())

    def touch(self) -> None:
        for access_layer in self._access_layers.values():
            access_layer.touch()

    def __iter__(self) -> Generator[str, None, None]:
        for name, access_layer in self._access_layers.items():
            for k in access_layer:
                yield name + self.KEY_SEPARATOR + k

    def __len__(self) -> int:
        return sum(map(len, self._access_layers))

    def __getitem__(self, key: str) -> VT:
        access_layer, key = self._parse(key)
        return access_layer[key]

    def __setitem__(self, key: str, value: VT) -> None:
        access_layer, key = self._parse(key)
        access_layer[key] = value

    def __delitem__(self, key: str) -> None:
        access_layer, key = self._parse(key)
        del access_layer[key]
