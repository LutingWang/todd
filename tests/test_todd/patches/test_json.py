import io
from pathlib import Path

from todd.patches.json import json_dump, json_load, jsonl_dump, jsonl_load


def test_json_dump(tmp_path: Path) -> None:
    object_ = dict(b=2, a=1)
    file = tmp_path / 'data.json'

    json_dump(object_, file, sort_keys=True)
    assert file.read_text() == '{"a":1,"b":2}'

    f = io.StringIO()
    json_dump(object_, f, compact=False)
    assert f.getvalue() == '{"b": 2, "a": 1}'


def test_json_load(tmp_path: Path) -> None:
    content = '{"a":1}'
    file = tmp_path / 'data.json'
    file.write_text(content)
    expected = dict(a=1)

    assert json_load(file) == expected
    assert json_load(io.StringIO(content)) == expected


def test_jsonl_dump(tmp_path: Path) -> None:
    objects = [
        dict(b=2, a=1),
        [1, 2],
    ]
    file = tmp_path / 'data.jsonl'

    jsonl_dump(objects, file, sort_keys=True)
    assert file.read_text() == '{"a":1,"b":2}\n[1,2]\n'

    f = io.StringIO()
    jsonl_dump(objects, f, compact=False)
    assert f.getvalue() == '{"b": 2, "a": 1}\n[1, 2]\n'


def test_jsonl_load(tmp_path: Path) -> None:
    content = '{"a":1}\n[1,2]\nnull\n'
    file = tmp_path / 'data.jsonl'
    file.write_text(content)
    expected = [dict(a=1), [1, 2], None]

    assert jsonl_load(file) == expected
    assert jsonl_load(io.StringIO(content)) == expected
