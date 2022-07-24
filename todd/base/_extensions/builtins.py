import contextlib
from typing import Generator

__all__ = [
    'setattr_temp',
]


@contextlib.contextmanager
def setattr_temp(obj, attr: str, value) -> Generator[None, None, None]:
    if hasattr(obj, attr):
        prev = getattr(obj, attr)
        setattr(obj, attr, value)
        yield
        setattr(obj, attr, prev)
    else:
        setattr(obj, attr, value)
        yield
        delattr(obj, attr)
