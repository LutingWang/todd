# Modified from
# https://github.com/open-mmlab/mmcv/blob/v1.6.1/mmcv/utils/config.py
#
# Copyright (c) OpenMMLab. All rights reserved.

__all__ = [
    'Config',
]

import pathlib
from collections import UserDict
from functools import reduce
from typing import Iterable, Mapping, Optional

import yapf.yapflib.yapf_api as yapf

BASE = '_base_'
DELETE = '_delete_'


class Config(UserDict):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__['_privilege'] = False

    def __setitem__(self, key, item) -> None:
        if isinstance(item, Mapping):
            item = self.__class__(item)
        super().__setitem__(key, item)

    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if self.has_privilege or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        if self.has_privilege:
            super().__delattr__(name)
        else:
            del self[name]

    @property
    def has_privilege(self) -> bool:
        return self.__dict__.get('_privilege', True)

    @classmethod
    def merge(cls, a, b):
        if not isinstance(b, Mapping):
            return b

        b = Config(b).copy()
        if b.pop(DELETE, False):
            return b
        if not isinstance(a, Mapping):
            return b

        a = Config(a).copy()
        for k in a.keys() & b.keys():
            b[k] = cls.merge(a[k], b[k])
        a.update(b)
        return a

    @classmethod
    def loads(cls, s: str, globals: Optional[dict] = None) -> 'Config':
        if globals is None:
            globals = dict()
        globals.setdefault('__name__', '__main__')

        config = cls()
        exec(s, globals, config)
        return config

    @classmethod
    def load(cls, file) -> 'Config':
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
            if isinstance(obj, (dict, Config)):
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
