import builtins
import functools
from types import FunctionType
from typing import Any, Callable

from .._extensions import get_logger

try:
    import ipdb

    get_logger().info("`ipdb` is installed. Using it for debugging.")
    builtins.breakpoint = ipdb.set_trace
except ImportError:
    pass


def patch(name: str) -> Callable[[FunctionType], FunctionType]:
    patched_func = getattr(builtins, name)

    def patcher(func: FunctionType) -> FunctionType:

        @functools.wraps(patched_func)
        def patch_func(obj, attr: str, *args, **kwargs):
            if '.' not in attr:
                return patched_func(obj, attr, *args, **kwargs)
            return func(obj, attr, *args, **kwargs)

        setattr(builtins, name, patch_func)
        return func

    return patcher


@patch('getattr')
def _getattr(obj: Any, attr: str, default=...):
    try:
        return eval('obj' + attr)
    except Exception:
        if default is not ...:
            return default
        raise


@patch('hasattr')
def _hasattr(obj: Any, attr: str) -> bool:
    default = object()
    return getattr(obj, attr, default) is not default


@patch('setattr')
def _setattr(obj, attr: str, value) -> None:
    attr, name = attr.rsplit('.', 1)
    if attr:
        obj = getattr(obj, attr)
    setattr(obj, name, value)


@patch('delattr')
def _delattr(obj, attr: str) -> None:
    attr, name = attr.rsplit('.', 1)
    if attr:
        obj = getattr(obj, attr)
    delattr(obj, name)
