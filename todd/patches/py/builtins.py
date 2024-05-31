__all__ = [
    'get_',
    'has_',
    'set_',
    'del_',
    'exec_',
    'map_',
    'classproperty',
    'descendant_classes',
    'NonInstantiableMeta',
]

import builtins
from typing import Any, Callable, NoReturn

try:
    import ipdb
    builtins.breakpoint = ipdb.set_trace
except ImportError:
    pass


def get_(obj, attr: str, default=...):
    try:
        return eval('__o' + attr, dict(__o=obj))  # nosec B307
    except Exception:
        if default is not ...:
            return default
        raise


def has_(obj, name: str) -> bool:
    """Check if an object has an attribute.

    Args:
        obj: The object to check.
        name: The attribute name.

    Returns:
        Whether the object has the attribute.
    """
    default = object()
    return get_(obj, name, default) is not default


def set_(obj, attr: str, value) -> None:
    """Set an attribute of an object.

    Args:
        obj: The object to set the attribute on.
        attr: The attribute name.
        value: The value to set.

    Raises:
        ValueError: If the attribute is invalid.
    """
    locals_: dict[str, Any] = dict()
    exec(f'__o{attr} = __v', dict(__o=obj, __v=value), locals_)  # nosec B102
    if len(locals_) != 0:
        raise ValueError(f"{attr} is invalid. Consider prepending a dot.")


def del_(obj, attr: str) -> None:
    """Delete an attribute of an object.

    Args:
        obj: The object to delete the attribute from.
        attr: The attribute name.
    """
    exec(f'del __o{attr}', dict(__o=obj))  # nosec B102


def exec_(source: str, **kwargs) -> dict[str, Any]:
    locals_: dict[str, Any] = dict()
    exec(source, kwargs, locals_)  # nosec B102
    return locals_


def map_(data, f: Callable[[Any], Any]):
    if isinstance(data, (list, tuple, set)):
        return data.__class__(map_(v, f) for v in data)
    if isinstance(data, dict):
        return data.__class__({k: map_(v, f) for k, v in data.items()})
    return f(data)


class classproperty:  # noqa: N801 pylint: disable=invalid-name

    def __init__(self, method: Callable[[Any], Any]) -> None:
        self._method = method

    def __get__(self, instance, cls):
        return self._method(cls)


def descendant_classes(cls: type) -> list[type]:
    classes = []
    for subclass in cls.__subclasses__():
        classes.append(subclass)
        classes.extend(descendant_classes(subclass))
    return classes


class NonInstantiableMeta(type):

    def __call__(cls, *args, **kwargs) -> NoReturn:
        raise RuntimeError(f"{cls.__name__} is instantiated")
