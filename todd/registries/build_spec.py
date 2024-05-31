__all__ = [
    'BuildSpec',
    'BuildSpecMixin',
]

from collections import UserDict
from typing import TYPE_CHECKING, Any, Callable, Generator

from ..patches.py import classproperty

if TYPE_CHECKING:
    from ..configs import Config

F = Callable[['Config'], Any]


class BuildSpec(UserDict[str, F]):
    """A class representing a build specification.

    A build specification is a mapping from keys to functions:

        >>> build_spec = BuildSpec(age=lambda c: c.value)
        >>> build_spec
        {'age': <function ...>}

    If a key of the build specification appears in the given configuration and
    the corresponding value is a `Config` object, the function will be applied
    to the value:

        >>> from todd import Config
        >>> config = Config(age=Config(value=3))
        >>> build_spec(config)
        {'age': 3}

    If the corresponding value is not a `Config` object, the function will not
    be applied:

        >>> config = Config(age='4')
        >>> build_spec(config)
        {'age': '4'}

    Keys of the build specification can be prefixed with an asterisk to
    indicate that the expected value is a collection of `Config` objects:

        >>> build_spec = BuildSpec({'*friends': lambda c: c.name})
        >>> build_spec
        {'*friends': <function ...>}

    If a key is prefixed with an asterisk and the corresponding value is a
    collection of `Config` objects, the function will be applied to each
    element:

        >>> config = Config(friends=[Config(name='Alice'), Config(name='Bob')])
        >>> build_spec(config)
        {'friends': ('Alice', 'Bob')}

    If any of the elements is not a `Config` object, the function will not be
    applied:

        >>> config = Config(friends=[Config(name='Alice'), 'Bob'])
        >>> build_spec(config)
        {'friends': [{'name': 'Alice'}, 'Bob']}
    """

    def build(self, f: F, v: Any, star: bool) -> Any:
        from ..configs import Config
        from ..utils import TreeUtil
        if not star:
            return f(v) if isinstance(v, Config) else v
        util = TreeUtil.get_util(v)
        assert util is not None  # user makes sure v is a collection
        if not all(isinstance(e, Config) for e in util.elements(v)):
            return v
        return util.map(f, v)

    def _items(self) -> Generator[tuple[str, F, bool], None, None]:
        for k, v in self.items():
            yield k.removeprefix('*'), v, k.startswith('*')

    def __call__(self, config: 'Config') -> 'Config':
        return config | {
            k: self.build(f, config[k], star)
            for k, f, star in self._items()
            if k in config
        }


class BuildSpecMixin:

    @classproperty
    def build_spec(self) -> BuildSpec:
        return BuildSpec()
