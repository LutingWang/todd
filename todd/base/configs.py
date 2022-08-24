# Modified from
# https://github.com/open-mmlab/mmcv/blob/v1.6.1/mmcv/utils/config.py
#
# Copyright (c) OpenMMLab. All rights reserved.

__all__ = [
    'Config',
]

import pathlib
from functools import reduce
from typing import Any, Dict, Iterable, Mapping, Optional, Type, TypeVar

import addict
import yapf.yapflib.yapf_api as yapf

T = TypeVar('T', bound='Config')

BASE = '_base_'
DELETE = '_delete_'


class Config(addict.Dict):

    @classmethod
    def merge(cls, a, b):
        if not isinstance(b, Mapping):
            return b

        b = cls(b).copy()
        if b.pop(DELETE, False):
            return b
        if not isinstance(a, Mapping):
            return b

        a = cls(a).copy()
        for k in a.keys() & b.keys():
            b[k] = cls.merge(a[k], b[k])
        a.update(b)
        return a

    @classmethod
    def loads(cls: Type[T], s: str, globals: Optional[dict] = None) -> T:
        if globals is None:
            globals = dict()
        globals.setdefault('__name__', '__main__')

        config: Dict[str, Any] = dict()
        exec(s, globals, config)
        return cls(config)

    @classmethod
    def load(cls: Type[T], file) -> T:
        file = pathlib.Path(file)
        file = file.resolve()

        config = cls.loads(file.read_text())
        configs = [
            cls.load(file.parent / base) for base in config.pop(BASE, [])
        ]
        configs.append(config)
        return reduce(cls.merge, configs)

    def dumps(self) -> str:

        def format(obj) -> str:
            contents: Iterable[str]
            if isinstance(obj, dict):
                if all(isinstance(k, str) and k.isidentifier() for k in obj):
                    contents = [k + '=' + format(v) for k, v in obj.items()]
                    delimiters = ('dict(', ')')
                else:
                    contents = [
                        format(k) + ': ' + format(v) for k, v in obj.items()
                    ]
                    delimiters = ('{', '}')
            elif isinstance(obj, list):
                contents = map(format, obj)
                delimiters = ('[', ']')
            elif isinstance(obj, tuple):
                contents = map(format, obj)
                delimiters = ('(', ')')
            elif isinstance(obj, set):
                contents = map(format, obj)
                delimiters = ('{', '}')
            else:
                return repr(obj)
            contents = sorted(contents)
            if len(obj) != 1:
                contents.append('')
            return delimiters[0] + ','.join(contents) + delimiters[1]

        assert all(isinstance(k, str) for k in self)
        code = '\n'.join(k + ' = ' + format(self[k]) for k in sorted(self))
        code, _ = yapf.FormatCode(code, verify=True)
        return code

    def dump(self, file) -> None:
        file = pathlib.Path(file)
        file = file.resolve()
        file.write_text(self.dumps())
