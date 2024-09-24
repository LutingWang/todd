__all__ = [
    'PyConfig',
]

from typing import Any

import yapf.yapflib.yapf_api as yapf

from ..bases.configs import Config
from ..patches.py_ import exec_
from ..registries import ConfigRegistry
from .serialize import SerializeMixin


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
        return exec_(__s, _kwargs_=kwargs)

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
        s = '\n'.join(f'{k}={self[k]!r}' for k in sorted(self))
        s, _ = yapf.FormatCode(s)
        return s
