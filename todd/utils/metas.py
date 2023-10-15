__all__ = [
    'NonInstantiableMeta',
]

from typing import NoReturn


class NonInstantiableMeta(type):

    def __call__(cls, *args, **kwargs) -> NoReturn:
        raise NotImplementedError
