__all__ = [
    'NormRegistry',
    'RegistryMeta',
    'Registry',
]

from collections import UserDict
from typing import Callable, Iterable, NoReturn, TypeVar

import torch.nn as nn

from .configs import Config
from .loggers import get_logger
from .misc import NonInstantiableMeta

T = TypeVar('T', bound=Callable)


class RegistryMeta(UserDict, NonInstantiableMeta):  # type: ignore[misc]
    """Meta class for registries.

    Under the hood, registries are simply dictionaries:

        >>> class Cat(metaclass=RegistryMeta): pass
        >>> class BritishShorthair: pass
        >>> Cat['british shorthair'] = BritishShorthair
        >>> Cat['british shorthair']
        <class '...BritishShorthair'>

    Users can also access registries via higher level APIs, i.e. `register`
    and `build`, for convenience.

    Registries can be subclassed.
    Derived classes of a registry are child registries:

        >>> class HairlessCat(Cat): pass
        >>> Cat.child('HairlessCat')
        <HairlessCat >
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize."""
        UserDict.__init__(self)
        NonInstantiableMeta.__init__(self, *args, **kwargs)
        self._logger = get_logger()

    def __call__(self, *args, **kwargs) -> NoReturn:
        """Prevent `Registry` classes from being initialized.

        Raises:
            TypeError: always.
        """
        raise TypeError("Registry cannot be initialized")

    def __missing__(self, key: str) -> NoReturn:
        """Missing key.

        Args:
            key: the missing key.

        Raises:
            KeyError: always.
        """
        self._logger.error(f"{key} does not exist in {self.__name__}")
        raise KeyError(key)

    def __repr__(self) -> str:
        return (
            f"<{self.__name__} "
            f"{' '.join(f'{k}={v}' for k, v in self.items())}>"
        )

    def __setitem__(self, key: str, item, forced: bool = False) -> None:
        """Register ``item`` with name ``key``.

        Args:
            key: name to be registered as.
            item: object to be registered.
            forced: if set, ``item`` will always be registered.
                By default, ``item`` will only be registered if ``key`` is
                not registered yet.

        Raises:
            KeyError: if ``key`` is already registered and ``forced`` is not
                set.

        By default, registries refuse to alter the registered object, in order
        to prevent unintended name clashes:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> Cat['british shorthair'] = 'british shorthair'
            >>> Cat['british shorthair'] = 'BritishShorthair'
            Traceback (most recent call last):
                ...
            KeyError: 'british shorthair'

        Specify the ``forced`` option to force registration:

            >>> Cat.__setitem__(
            ...     'british shorthair',
            ...     'BritishShorthair',
            ...     forced=True,
            ... )
            >>> Cat['british shorthair']
            'BritishShorthair'
        """
        if not forced and key in self:
            self._logger.error(f"{key} already exist in {self.__name__}")
            raise KeyError(key)
        return super().__setitem__(key, item)

    def __subclasses__(
        self=...,  # type: ignore[assignment]
    ) -> list['RegistryMeta']:
        """Refer to `ABC subclassed by meta classes`_.

        Returns:
            Children registries.

        .. _ABC subclassed by meta classes:
           https://blog.csdn.net/LutingWang/article/details/128320057
        """
        if self is ...:
            return NonInstantiableMeta.__subclasses__(RegistryMeta)
        return super().__subclasses__()

    def child(self, key: str) -> 'RegistryMeta':
        """Get a direct or indirect derived child registry.

        Args:
            key: dot separated subclass names.

        Raises:
            ValueError: if zero or multiple children are found.

        Returns:
            The derived registry.
        """
        for child_name in key.split('.'):
            subclasses = tuple(
                subclass
                for subclass in self.__subclasses__()  # type: ignore[misc]
                if subclass.__name__ == child_name
            )
            if len(subclasses) == 0:
                raise ValueError(f"{key} is not a child of {self}")
            if len(subclasses) > 1:
                raise ValueError(f"{key} matches multiple children of {self}")
            self, = subclasses
        return self

    def _parse(self, key: str) -> tuple['RegistryMeta', str]:
        """Parse the child name from the ``key``.

        Returns:
            The child registry and the name to be registered.
        """
        if '.' not in key:
            return self, key
        child_name, key = key.rsplit('.', 1)
        return self.child(child_name), key

    def register(
        self,
        keys: Iterable[str] = ...,  # type: ignore[assignment]
        **kwargs,
    ) -> Callable[[T], T]:
        """Register decorator.

        Args:
            keys: names to be registered as.
            kwargs: refer to `__setitem__`.

        Returns:
            Wrapper function.

        `register` is designed to be an decorator for classes and functions:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> @Cat.register()
            ... class Munchkin: pass
            >>> @Cat.register()
            ... def munchkin() -> str:
            ...     return 'munchkin'

        `register` has the following advantages:

        - default name

          By default, `register` uses the name of the registered object as
          ``keys``:

          >>> Cat['Munchkin']
          <class '...Munchkin'>
          >>> Cat['munchkin']
          <function munchkin at ...>

        - multiple names

          >>> @Cat.register(('British Longhair', 'british longhair'))
          ... class BritishLonghair: pass
          >>> 'British Longhair' in Cat
          True
          >>> 'british longhair' in Cat
          True

        - compatibility with child registries

          >>> class HairlessCat(Cat): pass
          >>> @Cat.register(('HairlessCat.CanadianHairless',))
          ... def canadian_hairless() -> str:
          ...     return 'canadian hairless'
          >>> HairlessCat
          <HairlessCat CanadianHairless=<function canadian_hairless at ...>>
        """

        def wrapper_func(obj: T) -> T:
            if keys is ...:
                self.__setitem__(obj.__name__, obj, **kwargs)
            else:
                for key in keys:
                    registry, key = self._parse(key)
                    registry.__setitem__(key, obj, **kwargs)
            return obj

        return wrapper_func

    def _build(self, config: Config):
        """Build an instance according to the given config.

        Args:
            config: instance specification.

        Returns:
            The built instance.

        To customize the build process of instances, registries must overload
        `_build` with a class method:

            >>> class Cat(metaclass=RegistryMeta):
            ...     @classmethod
            ...     def _build(cls, config: Config) -> str:
            ...         obj = RegistryMeta._build(cls, config)
            ...         return obj.__class__.__name__.upper()
            >>> @Cat.register()
            ... class Munchkin: pass
            >>> Cat.build(dict(type='Munchkin'))
            'MUNCHKIN'
        """
        type_ = self[config.pop('type')]
        return type_(**config)

    def build(
        self,
        config: Config,
        default_config: Config | None = None,
    ):
        """Call the registered object to construct a new instance.

        Args:
            config: build parameters.
            default_config: default configuration.

        Returns:
            The built instance.

        If the registered object is callable, the `build` method will
        automatically call the objects:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> @Cat.register()
            ... def tabby(name: str) -> str:
            ...     return f'Tabby {name}'
            >>> Cat.build(dict(type='tabby', name='Garfield'))
            'Tabby Garfield'

        Typically, ``config`` is a `Mapping` object.
        The ``type`` entry of ``config`` specifies the name of the registered
        object to be built.
        The other entries of ``config`` will be passed to the object's call
        method.

        ``default_config`` is the default configuration:

            >>> Cat.build(
            ...     dict(type='tabby'),
            ...     default_config=dict(name='Garfield'),
            ... )
            'Tabby Garfield'

        Refer to `_build` for customizations.
        """
        default_config = (
            Config() if default_config is None else Config(default_config)
        )
        default_config.update(config)

        config = default_config.copy()
        registry, config.type = self._parse(config.type)

        try:
            return registry._build(config)
        except Exception as e:
            # config may be altered
            self._logger.error(f'Failed to build {default_config}')
            raise e


class Registry(metaclass=RegistryMeta):
    """Base registry.

    To create custom registry, inherit from the `Registry` class:

        >>> class CatRegistry(Registry): pass
    """


class NormRegistry(Registry):
    pass


NormRegistry['BN1d'] = nn.BatchNorm1d
NormRegistry['BN2d'] = NormRegistry['BN'] = nn.BatchNorm2d
NormRegistry['BN3d'] = nn.BatchNorm3d
NormRegistry['SyncBN'] = nn.SyncBatchNorm
NormRegistry['GN'] = nn.GroupNorm
NormRegistry['LN'] = nn.LayerNorm
NormRegistry['IN1d'] = nn.InstanceNorm1d
NormRegistry['IN2d'] = NormRegistry['IN'] = nn.InstanceNorm2d
NormRegistry['IN3d'] = nn.InstanceNorm3d
