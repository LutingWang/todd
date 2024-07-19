__all__ = [
    'PyConfig',
]

import importlib
from typing import Any

import yapf.yapflib.yapf_api as yapf

from ..bases.configs import Config
from ..patches.py import exec_
from ..registries import ConfigRegistry
from .serialize import SerializeMixin


class _import_:  # noqa: N801 pylint: disable=invalid-name

    def __init__(self, name: str) -> None:
        self.__name = name
        self.__module = importlib.import_module(name)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.__module, attr)

    def __repr__(self) -> str:
        return f"_import_({self.__name!r})"


@ConfigRegistry.register_('py')
class PyConfig(SerializeMixin, Config):  # type: ignore[misc]

    @classmethod
    def _loads(cls, __s: str, **kwargs) -> dict[str, Any]:
        r"""Load config from string.

        Args:
            s: config string.

        Returns:
            The corresponding config.

        Config strings are valid python codes:

            >>> PyConfig._loads('a = 1\nb = dict(c=3)')
            {'a': 1, 'b': {'c': 3}}
        """
        return exec_(__s, PyConfig=cls, _import_=_import_, **kwargs)

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
            item = f'{k}={v!r}'
            items.append(item)
        items = imports + others
        code, _ = yapf.FormatCode('\n'.join(items))
        return code
