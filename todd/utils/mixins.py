__all__ = [
    'StateDictMixin',
    'HolderMixin',
]

import weakref
from typing import Any, Generic, Mapping, TypeVar, cast

T = TypeVar('T')


class StateDictMixin:

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return dict()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        pass


class HolderMixin(Generic[T]):

    def __init__(self, *args, instance: T, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        instance_proxy = (
            instance if isinstance(instance, weakref.ProxyTypes) else
            weakref.proxy(instance)
        )
        self._instance = cast(T, instance_proxy)
