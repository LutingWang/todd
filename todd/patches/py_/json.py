__all__ = [
    'json_dump',
    'json_load',
]

import json
import pathlib
from typing import Any


def json_dump(obj: Any, f: Any, *, compact: bool = True, **kwargs) -> None:
    if compact:
        kwargs.setdefault('separators', (',', ':'))

    if isinstance(f, str):
        with open(f, 'w') as f_:
            json.dump(obj, f_, **kwargs)
    elif isinstance(f, pathlib.Path):
        with f.open('w') as f_:
            json.dump(obj, f_, **kwargs)
    else:
        json.dump(obj, f, **kwargs)


def json_load(f: Any, **kwargs) -> Any:
    if isinstance(f, str):
        with open(f) as f_:
            return json.load(f_, **kwargs)
    if isinstance(f, pathlib.Path):
        with f.open() as f_:
            return json.load(f_, **kwargs)
    return json.load(f, **kwargs)
