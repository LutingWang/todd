__all__ = [
    'PartialRegistryMeta',
    'PartialRegistry',
]

from functools import partial
from typing import TYPE_CHECKING, no_type_check

from .registry import Any, Item, NonInstantiableMeta, RegistryMeta

if TYPE_CHECKING:
    from ..configs import Config


class PartialRegistryMeta(RegistryMeta):

    @no_type_check
    def __subclasses__(cls=...):
        if cls is ...:
            return NonInstantiableMeta.__subclasses__(PartialRegistryMeta)
        return super().__subclasses__()

    def _build(cls, item: Item, config: 'Config') -> Any:
        return partial(item, **config)


class PartialRegistry(metaclass=PartialRegistryMeta):
    pass
