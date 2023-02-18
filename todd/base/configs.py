__all__ = [
    'AttrDict',
    'Config',
    'DictAction',
]

import argparse
import difflib
import pathlib
import tempfile
import webbrowser
from collections import UserDict
from typing import Any, Mapping, MutableMapping, Sequence, TypeVar, cast

import toml
import yapf.yapflib.yapf_api as yapf

from .constants import FileType
from .patches import exec_, set_

AttrDictType = TypeVar('AttrDictType', bound='AttrDict')
ConfigType = TypeVar('ConfigType', bound='Config')


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
            return super().__setattr__(name, value)
        self[name] = value

    def __getattr__(self, name: str):
        if name == 'data':  # triggered in `copy.deepcopy`
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(e)


class Config(AttrDict, dict):  # type: ignore[misc]

    def __setitem__(self, name: str, value) -> None:
        """Set item.

        Args:
            name: item name or ``_delete_``.
            value: item value.

        If ``name`` is ``_delete_`` and ``value`` evaluates to `True`, the
        current config is cleared:

            >>> config = Config(a=1)
            >>> config['_delete_'] = True
            >>> config
            {}

        If ``name`` exists in the current config and the value is a `Sequence`,
        setting the value to a `Mapping` will recursively update the value:

            >>> config = Config(a=[1, 2, 3])
            >>> config['a'] = {1: 20, 3: 40}
            >>> config
            {'a': [1, 20, 3, 40]}
        """
        if name == '_delete_':
            if value:
                self.clear()
            return

        if isinstance(self.get(name), Sequence) and isinstance(value, Mapping):
            old_value = self.__class__(enumerate(self[name]))
            old_value.update(value)
            value = self[name].__class__(
                old_value[k] for k in range(len(old_value))
            )

        super().__setitem__(name, value)

    @classmethod
    def loads(
        cls: type[ConfigType],
        s: str,
        *,
        file_type: FileType = FileType.PYTHON,
        override: Mapping[str, Any] | None = None,
    ) -> ConfigType:
        """Load config from string.

        Args:
            s: config string.
            file_type: type of the contents of ``s``.
            override: fields to be overridden.

        Returns:
            The corresponding config.

        Raises:
            ValueError: if ``file_type`` is not supported.

        Config strings are valid python codes:

            >>> Config.loads('a = 1\\nb = dict(c=3)')
            {'a': 1, 'b': {'c': 3}}
        """
        if file_type is FileType.PYTHON:
            contents = exec_(s)
        elif file_type is FileType.TOML:
            contents = toml.loads(s)
        else:
            raise ValueError(f"Unsupported file type {file_type}")
        config = cls(contents)
        if override is not None:
            for k, v in override.items():
                set_(config, k, v)
        return config

    @classmethod
    def load(cls: type[ConfigType], file, **kwargs) -> ConfigType:
        """Load a configuration file.

        Args:
            file: any type of file path that can be converted to ``Path``.

        Returns:
            The loaded configuration.

        .. note:
            `file` need to be named with appropriate extension, so that the
            file type can be determined.
        """
        file = pathlib.Path(file)
        file_type = FileType(file.suffix)
        config = cls.loads(file.read_text(), file_type=file_type, **kwargs)
        base_config = cls()
        for base in config.pop('_base_', []):
            base_config.update(cls.load(file.parent / base))
        base_config.update(config)
        return base_config

    def dumps(self, *, file_type: FileType = FileType.PYTHON) -> str:
        """Reverse of `loads`.

        Args:
            file_type: type to be dumped.

        Returns:
            The corresponding config string.

        Raises:
            ValueError: if ``file_type`` is not supported.

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
        if file_type is FileType.PYTHON:
            config, _ = yapf.FormatCode(
                '\n'.join(f'{k}={repr(self[k])}' for k in sorted(self)),
                verify=True,
            )
        elif file_type is FileType.TOML:
            config = toml.dumps(self)
        else:
            raise ValueError(f"Unsupported file type {file_type}")
        return config

    def dump(self, file, **kwargs) -> None:
        """Dump the config to a file.

        Args:
            file: the file path.

        Refer to `dumps` for more details:

            >>> with tempfile.NamedTemporaryFile('r', suffix='.py') as f:
            ...     Config(a=1, b=dict(c=3)).dump(f.name)
            ...     f.readlines()
            ['a = 1\\n', "b = {'c': 3}\\n"]
        """
        file = pathlib.Path(file)
        file_type = FileType(file.suffix)
        file.write_text(self.dumps(file_type=file_type, **kwargs))

    def diff(
        self,
        other: ConfigType,
        file_type: FileType = FileType.TEXT,
    ) -> str:
        """Diff configs.

        Args:
            other: the other config to diff.
            file_type: diff text type.

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
        if file_type is FileType.TEXT:
            return '\n'.join(difflib.Differ().compare(a, b))
        if file_type is FileType.HTML:
            return difflib.HtmlDiff().make_file(a, b)
        raise ValueError(f"Unsupported file type {file_type}")

    def update(self, *args, **kwargs) -> None:
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
    parser.add_argument(
        '-f',
        '--file',
        default='terminal.txt',
        help=(
            "Path to output the diff results. Default is the stdout stream. "
            "To view the diff results in web browser, pass in "
            "`webbrowser.html`."
        ),
    )
    args = parser.parse_args()

    a: Config = args.a
    b: Config = args.b
    file = pathlib.Path(args.file)

    diff = a.diff(b, file_type=FileType(file.suffix))
    if file.stem == 'terminal':
        print(diff)
    elif file.stem == 'browser':
        with tempfile.NamedTemporaryFile(
            suffix=str(file),
            delete=False,
        ) as html_file:
            html_file.write(diff.encode('utf-8'))
            webbrowser.open('file://' + html_file.name)
    else:
        with open(file, 'w') as f:
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
            v = v[1:] if v.startswith(':') else eval(v)
            value_dict[k] = v
        setattr(namespace, self.dest, value_dict)
