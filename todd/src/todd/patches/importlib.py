__all__ = [
    'import_module',
]

import importlib
from types import ModuleType


def import_module(name: str, *args, **kwargs) -> ModuleType | None:
    try:
        return importlib.import_module(name, *args, **kwargs)
    except ImportError:
        return None
