__all__ = [
    'Config',
]

from typing import Any, Mapping, MutableMapping
from typing_extensions import Self

from ...patches.py_ import AttrDict, set_


class Config(AttrDict, dict[Any, Any]):  # type: ignore[misc]

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
        if not self:
            super().__setitem__(name, value)
            return
        if name == '_delete_':
            if value:
                self.clear()
            return
        if name == '_override_':
            self.override(value)
            return
        super().__setitem__(name, value)

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

    def get_config(self, key: str) -> Self:
        if key in self:
            config = self[key]
            assert isinstance(config, self.__class__)
        else:
            config = self.__class__()
            self[key] = config
        return config
