__all__ = [
    'NonInstantiableMeta',
]

from typing import NoReturn


class NonInstantiableMeta(type):

    def __call__(cls, *args, **kwargs) -> NoReturn:
        raise RuntimeError(f"{cls.__name__} is instantiated")
