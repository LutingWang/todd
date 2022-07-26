__all__ = [
    'getattr_recur',
    'hasattr_recur',
    'setattr_recur',
    'delattr_recur',
    'setattr_temp',
]

import contextlib
import functools
from typing import Any, Generator


def patch(patched_func):

    def patcher(func):

        @functools.wraps(func)
        def patch_func(obj, attr, *args, **kwargs):
            if '.' not in attr:
                func = patched_func
            return func(obj, attr, *args, **kwargs)

        return patch_func

    return patcher


@patch(getattr)
def getattr_recur(obj: Any, attr: str, default=...):
    try:
        return eval('obj' + attr)
    except Exception:
        if default is not ...:
            return default
        raise


@patch(hasattr)
def hasattr_recur(obj: Any, attr: str) -> bool:
    default = object()
    return getattr_recur(obj, attr, default) is not default


@patch(setattr)
def setattr_recur(obj, attr: str, value) -> None:
    attr, name = attr.rsplit('.', 1)
    if attr:
        obj = getattr_recur(obj, attr)
    setattr(obj, name, value)


@patch(delattr)
def delattr_recur(obj, attr: str) -> None:
    attr, name = attr.rsplit('.', 1)
    if attr:
        obj = getattr_recur(obj, attr)
    delattr(obj, name)


@contextlib.contextmanager
def setattr_temp(obj, attr: str, value) -> Generator[None, None, None]:
    if hasattr_recur(obj, attr):
        prev = getattr_recur(obj, attr)
        setattr_recur(obj, attr, value)
        yield
        setattr_recur(obj, attr, prev)
    else:
        setattr_recur(obj, attr, value)
        yield
        delattr_recur(obj, attr)
