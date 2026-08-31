__all__ = [
    'get_named_classes',
    'get_classes',
]

import inspect
from types import ModuleType
from typing import Any


def get_named_classes(module: ModuleType, *args) -> dict[str, type[Any]]:
    named_classes = dict(inspect.getmembers(module, inspect.isclass))
    if args:
        named_classes = {
            k: v
            for k, v in named_classes.items()
            if issubclass(v, args)
        }
    return named_classes


def get_classes(*args, **kwargs) -> set[type[Any]]:
    return set(get_named_classes(*args, **kwargs).values())
