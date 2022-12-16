# Modified from
# https://github.com/open-mmlab/mmcv/blob/v1.6.1/mmcv/utils/config.py
#
# Copyright (c) OpenMMLab. All rights reserved.

__all__ = [
    'Config',
    'DictAction',
]

import argparse
import difflib
import pathlib
import tempfile
import webbrowser
from functools import reduce
from typing import Any, Iterable, Mapping, NoReturn, Sequence, TypeVar

import addict
import yapf.yapflib.yapf_api as yapf

T = TypeVar('T', bound='Config')

BASE = '_base_'
DELETE = '_delete_'


class Config(addict.Dict):

    def __missing__(self, name) -> NoReturn:
        raise KeyError(name)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except KeyError as e:
            raise AttributeError(e)

    @classmethod
    def merge(cls, a, b):
        if not isinstance(b, Mapping):
            return b

        b = cls(b)
        if isinstance(a, list) and all(isinstance(x, int) for x in b):
            a = list(a)
            while len(a) in b:
                a.append(b.pop(len(a)))
            for i in sorted(b):
                a[i] = (
                    cls.merge(a[i], b[i])
                    if isinstance(a[i], Mapping) else b[i]
                )
            return a

        if b.pop(DELETE, False):
            return b
        if not isinstance(a, Mapping):
            return b

        a = cls(a)
        for k in b:
            a[k] = cls.merge(a[k], b[k]) if k in a else b[k]
        return a

    @classmethod
    def loads(cls: type[T], s: str, globals: dict | None = None) -> T:
        if globals is None:
            globals = dict()
        globals.setdefault('__name__', '__main__')

        config: dict[str, Any] = dict()
        exec(s, globals, config)
        return cls(config)

    @classmethod
    def load(cls: type[T], file) -> T:
        file = pathlib.Path(file)
        file = file.resolve()

        config = cls.loads(file.read_text())
        configs = [
            cls.load(file.parent / base) for base in config.pop(BASE, [])
        ]
        configs.append(config)
        return reduce(cls.merge, configs)

    @classmethod
    def diff(
        cls: type[T],
        a: T,
        b: T,
        mode: str = 'text',
    ) -> str:
        a_ = a.dumps().split('\n')
        b_ = b.dumps().split('\n')
        if mode == 'text':
            return '\n'.join(difflib.Differ().compare(a_, b_))
        if mode == 'html':
            return difflib.HtmlDiff().make_file(a_, b_)
        raise ValueError(f"Invalid mode {mode}.")

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
                contents = sorted(contents)
            elif isinstance(obj, list):
                contents = map(format, obj)
                contents = list(contents)
                delimiters = ('[', ']')
            elif isinstance(obj, tuple):
                contents = map(format, obj)
                contents = list(contents)
                delimiters = ('(', ')')
            elif isinstance(obj, set):
                contents = map(format, obj)
                contents = sorted(contents)
                delimiters = ('{', '}')
            else:
                return repr(obj)
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


def diff_cli() -> None:
    parser = argparse.ArgumentParser(description="Compare Configs")
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--out')
    args = parser.parse_args()

    a = Config.load(args.a)
    b = Config.load(args.b)

    if args.out is None:
        diff_mode = 'text'
    elif args.out.endswith('.txt'):
        diff_mode = 'text'
    elif args.out.endswith('.html'):
        diff_mode = 'html'
    elif args.out == 'browser':
        diff_mode = 'html'
    else:
        raise ValueError(f"Unknown output mode: {args.out}.")

    diff = Config.diff(a, b, diff_mode)
    if args.out is None:
        print(diff)
    elif args.out == 'browser':
        with tempfile.NamedTemporaryFile(
            suffix='.html',
            delete=False,
        ) as html_file:
            html_file.write(diff.encode('utf-8'))
            webbrowser.open('file://' + html_file.name)
    else:
        with open(args.out, 'w') as f:
            f.write(diff)


class DictAction(argparse.Action):
    """``argparse`` action to parse arguments in the form of key-value pairs.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--dict', action=DictAction)
        DictAction(...)
        >>> parser.parse_args('--dict key1:"value1" key2:"value2"'.split())
        Namespace(dict={'key1': 'value1', 'key2': 'value2'})
    """

    def __init__(self, *args, nargs=None, **kwargs):
        """
        Args:
            nargs: The number of dictionary arguments that should be consumed.
        """
        if nargs not in [None, argparse.ZERO_OR_MORE]:
            raise ValueError(f"Invalid nargs={nargs}")

        super().__init__(
            *args,
            nargs=argparse.ZERO_OR_MORE,
            default=nargs and [],
            **kwargs,
        )
        self._append = bool(nargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if not isinstance(values, Sequence):
            raise ValueError(f'values must be a sequence, but got {values}')
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f'values must be strings, but got {values}')
        value_dict = Config()
        for value in values:
            k, v = value.split(':', 1)
            k = k.strip()
            v = v[1:] if v.startswith(':') else eval(v)
            value_dict[k] = v
        if self._append:
            value_dict_list = getattr(namespace, self.dest, [])
            value_dict_list.append(value_dict)
            setattr(namespace, self.dest, value_dict_list)
        else:
            setattr(namespace, self.dest, value_dict)
