__all__ = [
    'get_',
    'has_',
    'set_',
    'del_',
    'exec_',
    'map_',
    'set_temp',
]

import contextlib
from typing import Any, Callable, Generator


def get_(obj, attr: str, default=...):
    try:
        return eval('__o' + attr, dict(__o=obj))
    except Exception:
        if default is not ...:
            return default
        raise


def has_(obj, name: str) -> bool:
    default = object()
    return get_(obj, name, default) is not default


def set_(obj, attr: str, value) -> None:
    locals_: dict[str, Any] = dict()
    exec(f'__o{attr} = __v', dict(__o=obj, __v=value), locals_)
    if len(locals_) != 0:
        raise ValueError(f"{attr} is invalid. Consider prepending a dot.")


def del_(obj, attr: str) -> None:
    exec(f'del __o{attr}', dict(__o=obj))


def exec_(source: str, **kwargs) -> dict[str, Any]:
    locals_: dict[str, Any] = dict()
    exec(source, kwargs, locals_)
    return locals_


def map_(data, f: Callable[[Any], Any]):
    if isinstance(data, (list, tuple, set)):
        return data.__class__(map_(v, f) for v in data)
    if isinstance(data, dict):
        return data.__class__({k: map_(v, f) for k, v in data.items()})
    return f(data)


@contextlib.contextmanager
def set_temp(obj, name: str, value) -> Generator[None, None, None]:
    if has_(obj, name):
        prev = get_(obj, name)
        set_(obj, name, value)
        yield
        set_(obj, name, prev)
    else:
        set_(obj, name, value)
        yield
        del_(obj, name)
