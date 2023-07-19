__all__ = [
    'RegistryMeta',
    'Registry',
    'NormRegistry',
    'LrSchedulerRegistry',
    'OptimizerRegistry',
    'RunnerRegistry',
    'CallbackRegistry',
    'LossRegistry',
    'SchedulerRegistry',
    'VisualRegistry',
    'HookRegistry',
    'AccessLayerRegistry',
    'DatasetRegistry',
    'StrategyRegistry',
    'SamplerRegistry',
    'ModelRegistry',
]

import re
from collections import UserDict
from typing import Callable, Iterable, NoReturn, TypeVar

import torch
import torch.nn as nn
import torch.utils.data

from ..utils import NonInstantiableMeta
from .configs import Config
from .patches import logger

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
        logger.error(f"{key} does not exist in {self.__name__}")
        raise KeyError(key)

    def __repr__(self) -> str:
        items = ' '.join(f'{k}={v}' for k, v in self.items())
        return f"<{self.__name__} {items}>"

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
            logger.error(f"{key} already exist in {self.__name__}")
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
        keys: Iterable[str] | None = None,
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
            if keys is None:
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
            >>> Cat.build(Config(type='Munchkin'))
            'MUNCHKIN'
        """
        type_ = self[config.pop('type')]
        return type_(**config)

    def build(self, config: Config, **kwargs):
        """Call the registered object to construct a new instance.

        Args:
            config: build parameters.
            kwargs: default configuration.

        Returns:
            The built instance.

        If the registered object is callable, the `build` method will
        automatically call the objects:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> @Cat.register()
            ... def tabby(name: str) -> str:
            ...     return f'Tabby {name}'
            >>> Cat.build(Config(type='tabby', name='Garfield'))
            'Tabby Garfield'

        Typically, ``config`` is a `Mapping` object.
        The ``type`` entry of ``config`` specifies the name of the registered
        object to be built.
        The other entries of ``config`` will be passed to the object's call
        method.

        Keyword arguments are the default configuration:

            >>> Cat.build(
            ...     Config(type='tabby'),
            ...     name='Garfield',
            ... )
            'Tabby Garfield'

        Refer to `_build` for customizations.
        """
        default_config = Config(kwargs)
        default_config.update(config)

        config = default_config.copy()
        registry, config.type = self._parse(config.type)

        try:
            return registry._build(config)
        except Exception as e:
            # config may be altered
            logger.error(f"Failed to build\n{default_config.dumps()}")
            raise e


class Registry(metaclass=RegistryMeta):
    """Base registry.

    To create custom registry, inherit from the `Registry` class:

        >>> class CatRegistry(Registry): pass
    """


class NormRegistry(Registry):
    pass


class OptimizerRegistry(Registry):

    @staticmethod
    def params(model: nn.Module, config: Config) -> Config:
        config.params = [
            p for n, p in model.named_parameters()
            if re.match(config.params, n)
        ]
        return config

    @classmethod
    def _build(cls, config: Config) -> torch.optim.Optimizer:
        model: nn.Module = config.pop('model')
        params: Iterable[Config] | None = config.pop('params', None)
        config.params = model.parameters() if params is None else [
            cls.params(model, p) for p in params
        ]
        return RegistryMeta._build(cls, config)


class LrSchedulerRegistry(Registry):
    pass


class RunnerRegistry(Registry):
    pass


class CallbackRegistry(Registry):
    pass


class LossRegistry(Registry):
    pass


class SchedulerRegistry(Registry):
    pass


class VisualRegistry(Registry):
    pass


class HookRegistry(Registry):
    pass


class DatasetRegistry(Registry):
    pass


class AccessLayerRegistry(Registry):
    pass


class StrategyRegistry(Registry):
    pass


class SamplerRegistry(Registry):
    pass


class ModelRegistry(Registry):
    pass


for c in torch.utils.data.Sampler.__subclasses__():
    SamplerRegistry.register()(c)

for c in torch.optim.Optimizer.__subclasses__():
    OptimizerRegistry.register()(c)

for c in torch.optim.lr_scheduler.LRScheduler.__subclasses__():
    LrSchedulerRegistry.register()(c)

NormRegistry['BN1d'] = nn.BatchNorm1d
NormRegistry['BN2d'] = NormRegistry['BN'] = nn.BatchNorm2d
NormRegistry['BN3d'] = nn.BatchNorm3d
NormRegistry['SyncBN'] = nn.SyncBatchNorm
NormRegistry['GN'] = nn.GroupNorm
NormRegistry['LN'] = nn.LayerNorm
NormRegistry['IN1d'] = nn.InstanceNorm1d
NormRegistry['IN2d'] = NormRegistry['IN'] = nn.InstanceNorm2d
NormRegistry['IN3d'] = nn.InstanceNorm3d
