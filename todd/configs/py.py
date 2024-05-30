__all__ = [
    'PyConfig',
]

import importlib

import yapf.yapflib.yapf_api as yapf

from ..patches.py import exec_
from ..registries import ConfigRegistry
from .serializable import SerializableConfig


class _import_:  # noqa: N801 pylint: disable=invalid-name

    def __init__(self, name: str) -> None:
        self.__name = name
        self.__module = importlib.import_module(name)

    def __getattr__(self, attr: str):
        return getattr(self.__module, attr)

    def __repr__(self) -> str:
        return f"_import_({repr(self.__name)})"


@ConfigRegistry.register_('py')
class PyConfig(SerializableConfig):  # type: ignore[misc]

    @staticmethod
    def _loads(s: str) -> dict:
        r"""Load config from string.

        Args:
            s: config string.

        Returns:
            The corresponding config.

        Config strings are valid python codes:

            >>> PyConfig.loads('a = 1\nb = dict(c=3)')
            {'a': 1, 'b': {'c': 3}}
        """
        return exec_(s, _import_=_import_)

    def dumps(self) -> str:
        """Reverse of `loads`.

        Returns:
            The corresponding config string.

        The dumped string is a readable version of the config:

            >>> config = PyConfig(
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
