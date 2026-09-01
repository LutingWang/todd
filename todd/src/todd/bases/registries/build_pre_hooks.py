__all__ = [
    'BuildPreHookMixin',
]

from abc import abstractmethod

from ..configs import Config
from .base import Item, RegistryMeta


class BuildPreHookMixin:

    @classmethod
    @abstractmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        return config
