__all__ = [
    'json_dump',
    'json_load',
]

import json
from typing import Any


def json_dump(obj: Any, f: Any, *, compact: bool = True, **kwargs) -> None:
    if compact:
        kwargs.setdefault('separators', (',', ':'))

    if not isinstance(f, str):
        json.dump(obj, f, **kwargs)
        return

    with open(f, 'w') as f_:
        json.dump(obj, f_, **kwargs)


def json_load(f: Any, **kwargs) -> Any:
    if not isinstance(f, str):
        return json.load(f, **kwargs)

    with open(f) as f_:
        return json.load(f_, **kwargs)
