__all__ = [
    'json_dump',
    'json_load',
    'jsonl_dump',
    'jsonl_load',
]

import json
import pathlib
from collections.abc import Iterable
from typing import Any


def json_dump(object_: Any, f: Any, *, compact: bool = True, **kwargs) -> None:
    if isinstance(f, (str, pathlib.Path)):
        with open(f, 'w') as f_:
            json_dump(object_, f_, compact=compact, **kwargs)
        return

    if compact:
        kwargs.setdefault('separators', (',', ':'))
    json.dump(object_, f, **kwargs)


def json_load(f: Any, **kwargs) -> Any:
    if isinstance(f, (str, pathlib.Path)):
        with open(f) as f_:
            return json_load(f_, **kwargs)
    return json.load(f, **kwargs)


def jsonl_dump(
    objects: Iterable[Any],
    f: Any,
    *,
    compact: bool = True,
    **kwargs,
) -> None:
    if isinstance(f, (str, pathlib.Path)):
        with open(f, 'w') as f_:
            jsonl_dump(objects, f_, compact=compact, **kwargs)
        return

    if compact:
        kwargs.setdefault('separators', (',', ':'))
    for object_ in objects:
        json.dump(object_, f, **kwargs)
        f.write('\n')


def jsonl_load(f: Any, **kwargs) -> list[Any]:
    if isinstance(f, (str, pathlib.Path)):
        with open(f) as f_:
            return jsonl_load(f_, **kwargs)
    return [json.loads(line, **kwargs) for line in f]
