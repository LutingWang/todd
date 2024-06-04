__all__ = [
    'json_dump',
]

import json
from typing import Any

from .typing import SupportsWrite


def _json_dump(
    obj: Any,
    f: SupportsWrite,
    compact: bool = True,
    **kwargs,
) -> None:
    if compact:
        kwargs.setdefault('separators', (',', ':'))
    json.dump(obj, f, **kwargs)


def json_dump(obj: Any, f: Any, **kwargs) -> None:
    if isinstance(f, SupportsWrite):
        _json_dump(obj, f, **kwargs)
        return
    with open(f, 'w') as f_:
        _json_dump(obj, f_, **kwargs)
