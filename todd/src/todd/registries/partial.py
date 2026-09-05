__all__ = [
    'PartialRegistryMeta',
    'PartialRegistry',
]

from functools import partial
from typing import Any, no_type_check

from ..configs import Config
from ..patches import NonInstantiableMeta
from .registry import Item, RegistryMeta


class PartialRegistryMeta(RegistryMeta):

    @no_type_check
    def __subclasses__(cls=...):
        if cls is ...:
            return NonInstantiableMeta.__subclasses__(PartialRegistryMeta)
        return super().__subclasses__()

    def _build(cls, item: Item, config: Config) -> Any:
        return partial(item, **config)


class PartialRegistry(metaclass=PartialRegistryMeta):
    pass
