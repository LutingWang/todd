__all__ = [
    'HolderMixin',
]

import weakref
from typing import Generic, TypeVar, cast

T = TypeVar('T')


class HolderMixin(Generic[T]):

    def __init__(self, *args, instance: T | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if instance is not None:
            self.bind(instance)

    @property
    def holding(self) -> bool:
        return hasattr(self, '_instance')

    def bind(self, instance: T) -> None:
        assert not self.holding
        instance_proxy = (
            instance if isinstance(instance, weakref.ProxyTypes) else
            weakref.proxy(instance)
        )
        self._instance = cast(T, instance_proxy)
