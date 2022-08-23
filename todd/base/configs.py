# Modified from
# https://github.com/open-mmlab/mmcv/blob/v1.6.1/mmcv/utils/config.py
#
# Copyright (c) OpenMMLab. All rights reserved.

__all__ = [
    'Config',
]

import copy
import pathlib
from functools import reduce
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    NoReturn,
    Optional,
    TypeVar,
)

import yapf.yapflib.yapf_api as yapf

T = TypeVar('T', bound='Config')

BASE = '_base_'
DELETE = '_delete_'


# Do not inherit `UserDict`. Otherwise, config cannot include `data` entry.
class Config(MutableMapping):
    _data: dict

    def __init__(self, *args, **kwargs) -> None:
        self.__dict__.update(_data=dict())
        self.update(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._data)

    def __missing__(self, key) -> NoReturn:
        raise KeyError(key)

    def __getitem__(self, key):
        if key in self._data:
            return self._data[key]
        return self.__missing__(key)

    def __setitem__(self, key, item) -> None:
        if isinstance(item, Mapping):
            item = self.__class__(item)
        self._data[key] = item

    def __delitem__(self, key) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __contains__(self, key) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return repr(self._data)

    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __copy__(self: T) -> T:
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        inst._data = self._data.copy()
        return inst

    def __deepcopy__(self: T, memo: Dict[int, Any]) -> T:
        inst = self.__class__.__new__(self.__class__)
        memo[id(self)] = inst
        inst.__dict__.update({
            k: copy.deepcopy(v, memo)
            for k, v in self.__dict__.items()
        })
        return inst

    def __getstate__(self) -> dict:
        return self._data

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(_data=state)

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

    def copy(self: T) -> T:
        return self.__copy__()

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
