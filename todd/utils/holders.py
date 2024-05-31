__all__ = [
    'HolderMixin',
]

import weakref
from typing import Generic, TypeVar, cast

T = TypeVar('T')


class HolderMixin(Generic[T]):

    def __init__(self, *args, instance: T, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        instance_proxy = (
            instance if isinstance(instance, weakref.ProxyTypes) else
            weakref.proxy(instance)
        )
        self._instance = cast(T, instance_proxy)
