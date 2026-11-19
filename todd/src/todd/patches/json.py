__all__ = [
    'PythonJSONEncoder',
    'json_dump',
    'json_load',
    'jsonl_dump',
    'jsonl_load',
]

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class PythonJSONEncoder(json.JSONEncoder):

    def iterencode(self, o: Any, _one_shot: bool = False) -> Iterator[str]:
        return super().iterencode(o, _one_shot=False)


def json_dump(object_: Any, f: Any, *, compact: bool = True, **kwargs) -> None:
    if isinstance(f, (str, Path)):
        with open(f, 'w') as f_:
            json_dump(object_, f_, compact=compact, **kwargs)
        return

    if compact:
        kwargs.setdefault('separators', (',', ':'))
    json.dump(object_, f, **kwargs)


def json_load(f: Any, **kwargs) -> Any:
    if isinstance(f, (str, Path)):
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
    if isinstance(f, (str, Path)):
        with open(f, 'w') as f_:
            jsonl_dump(objects, f_, compact=compact, **kwargs)
        return

    if compact:
        kwargs.setdefault('separators', (',', ':'))
    for object_ in objects:
        json.dump(object_, f, **kwargs)
        f.write('\n')


def jsonl_load(f: Any, **kwargs) -> list[Any]:
    if isinstance(f, (str, Path)):
        with open(f) as f_:
            return jsonl_load(f_, **kwargs)
    return [json.loads(line, **kwargs) for line in f]
