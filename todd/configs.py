__all__ = [
    'AttrDict',
    'Config',
    'DictAction',
]

import argparse
import difflib
import importlib
import pathlib
import tempfile
import webbrowser
from collections import UserDict
from typing import Any, Mapping, MutableMapping, Sequence, cast
from typing_extensions import Self

import yapf.yapflib.yapf_api as yapf

from .patches import exec_, set_


class AttrDict(UserDict):

    @classmethod
    def _map(cls, item):
        if isinstance(item, (list, tuple, set)):
            return item.__class__(map(cls._map, item))
        if isinstance(item, dict):
            return cls(item)
        return item

    def __setitem__(self, name: str, value) -> None:
        value = self._map(value)
        super().__setitem__(name, value)

    def __setattr__(self, name: str, value) -> None:
        if name == 'data' or hasattr(self.__class__, name):
            super().__setattr__(name, value)
            return
        self[name] = value

    def __getattr__(self, name: str):
        if name == 'data':  # triggered in `copy.deepcopy`
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e) from e

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(e) from e


class _import_:  # noqa: N801 pylint: disable=invalid-name

    def __init__(self, name: str) -> None:
        self.__name = name
        self.__module = importlib.import_module(name)

    def __getattr__(self, attr: str):
        return getattr(self.__module, attr)

    def __repr__(self) -> str:
        return f"_import_({repr(self.__name)})"


class Config(AttrDict, dict):  # type: ignore[misc]

    def __setitem__(self, name: str, value) -> None:
        """Set item.

        Args:
            name: item name, ``_delete_``, or ``__override__``.
            value: item value.

        The default behavior is same as `AttrDict.__setitem__`:

            >>> config = Config(a=1)
            >>> config['b'] = 2
            >>> config
            {'a': 1, 'b': 2}

        If ``name`` is ``_delete_`` and ``value`` evaluates to `False`, nothing
        happens:

            >>> config = Config(a=1)
            >>> config['_delete_'] = False
            >>> config
            {'a': 1}

        If ``name`` is ``_delete_`` and ``value`` evaluates to `True`, the
        current config is cleared:

            >>> config = Config(a=1)
            >>> config['_delete_'] = True
            >>> config
            {}

        If ``name`` is ``_override_`` and the current config is not empty, the
        current config is updated with ``value``:

            >>> config = Config(a=1, b=[2, 3, 4])
            >>> config['_override_'] = {'.a': 2, '.b[0]': 5, '.c': 6}
            >>> config
            {'a': 2, 'b': [5, 3, 4], 'c': 6}

        If the current config is empty, all items are set as usual:

            >>> config = Config()
            >>> config['_delete_'] = True
            >>> config
            {'_delete_': True}
            >>> config = Config()
            >>> config['_override_'] = {'.a': 1}
            >>> config
            {'_override_': {'.a': 1}}
        """
        if self:
            if name == '_delete_':
                if value:
                    self.clear()
                return
            if name == '_override_':
                self.override(value)
                return
        super().__setitem__(name, value)

    @staticmethod
    def _loads(s: str) -> dict:
        return exec_(s, _import_=_import_)

    @classmethod
    def loads(cls, s: str) -> Self:
        r"""Load config from string.

        Args:
            s: config string.

        Returns:
            The corresponding config.

        Config strings are valid python codes:

            >>> Config.loads('a = 1\nb = dict(c=3)')
            {'a': 1, 'b': {'c': 3}}
        """
        return cls(cls._loads(s))

    @classmethod
    def load(cls, file) -> Self:
        file = pathlib.Path(file)
        # `loads` does not support `_delete_` with `_base_`
        config = cls._loads(file.read_text())
        base_config = cls()
        for base in config.pop('_base_', []):
            base_config.update(cls.load(file.parent / base))
        base_config.update(config)
        return base_config

    def dumps(self) -> str:
        """Reverse of `loads`.

        Returns:
            The corresponding config string.

        The dumped string is a readable version of the config:

            >>> config = Config(
            ...     a=1,
            ...     b=dict(c=3),
            ...     d={
            ...         5: 'e',
            ...         'f': ['g', ('h', 'i', 'j')],
            ...     },
            ...     k=[2, 1],
            ...     l='mn',
            ... )
            >>> print(config.dumps())
            a = 1
            b = {'c': 3}
            d = {5: 'e', 'f': ['g', ('h', 'i', 'j')]}
            k = [2, 1]
            l = 'mn'
            <BLANKLINE>
        """
        imports: list[str] = []
        others: list[str] = []
        for k in sorted(self):
            v = self[k]
            items = imports if isinstance(v, _import_) else others
            item = f'{k}={repr(v)}'
            items.append(item)
        items = imports + others
        code, _ = yapf.FormatCode('\n'.join(items))
        return code

    def dump(self, file) -> None:
        r"""Dump the config to a file.

        Args:
            file: the file path.

        Refer to `dumps` for more details.

        Example:
            >>> with tempfile.NamedTemporaryFile('r') as f:
            ...     Config(a=1, b=dict(c=3)).dump(f.name)
            ...     f.readlines()
            ['a = 1\n', "b = {'c': 3}\n"]
        """
        pathlib.Path(file).write_text(self.dumps())

    def diff(self, other: 'Config', html: bool = False) -> str:
        """Diff configs.

        Args:
            other: the other config to diff.
            html: output diff in html format. Default is pure text.

        Returns:
            Diff message.

        Diff the config strings:

            >>> a = Config(a=1)
            >>> b = Config(a=1, b=dict(c=3))
            >>> print(a.diff(b))
              a = 1
            + b = {'c': 3}
            <BLANKLINE>
        """
        a = self.dumps().split('\n')
        b = other.dumps().split('\n')
        if html:
            return difflib.HtmlDiff().make_file(a, b)
        return '\n'.join(difflib.Differ().compare(a, b))

    def override(self, other: Mapping[str, Any]) -> None:
        for k, v in other.items():
            set_(self, k, v)

    def update(  # pylint: disable=arguments-differ
        self,
        *args,
        **kwargs,
    ) -> None:
        for m in args + (kwargs, ):
            for k, v in dict(m).items():
                old_v = self.get(k)
                if (
                    isinstance(old_v, MutableMapping)
                    and isinstance(v, Mapping)
                ):
                    old_v.update(v)
                else:
                    self[k] = v


def diff_cli() -> None:
    parser = argparse.ArgumentParser(description="Compare Configs")
    parser.add_argument('a', type=Config.load)
    parser.add_argument('b', type=Config.load)
    parser.add_argument('--out', default='terminal')
    args = parser.parse_args()

    a: Config = args.a
    b: Config = args.b
    out: str = args.out

    diff = a.diff(b, out == 'browser' or out.endswith('.html'))
    if out == 'terminal':
        print(diff)
    elif out == 'browser':
        with tempfile.NamedTemporaryFile(
            suffix='.html',
            delete=False,
        ) as html_file:
            html_file.write(diff.encode('utf-8'))
            webbrowser.open('file://' + html_file.name)
    else:
        with open(out, 'w') as f:
            f.write(diff)


class DictAction(argparse.Action):
    """``argparse`` action to parse arguments in the form of key-value pairs.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--dict', action=DictAction)
        DictAction(...)
        >>> parser.parse_args('--dict key1::value1 key2::value2'.split())
        Namespace(dict={'key1': 'value1', 'key2': 'value2'})
    """

    def __init__(self, *args, **kwargs) -> None:
        assert 'nargs' not in kwargs
        kwargs['nargs'] = argparse.ZERO_OR_MORE
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence | None = None,
        option_string: str | None = None,
    ) -> None:
        values = cast(Sequence[str], values)
        value_dict: dict[str, Any] = dict()
        for value in values:
            k, v = value.split(':', 1)
            k = k.strip()
            v = v[1:] if v.startswith(':') else eval(v)  # nosec B307
            value_dict[k] = v
        setattr(namespace, self.dest, value_dict)
