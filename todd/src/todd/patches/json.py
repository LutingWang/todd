__all__ = [
    'python_json',
    'json_dump',
    'json_load',
    'jsonl_dump',
    'jsonl_load',
]

import json
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def python_json() -> Generator[None, None, None]:
    encoder = json.encoder.c_make_encoder  # type: ignore[attr-defined]
    json.encoder.c_make_encoder = None  # type: ignore[attr-defined]
    try:
        yield
    finally:
        json.encoder.c_make_encoder = encoder  # type: ignore[attr-defined]


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
