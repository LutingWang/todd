# pylint: disable=no-value-for-parameter

__all__ = [
    'Item',
    'RegistryMeta',
    'PartialRegistryMeta',
    'Registry',
    'PartialRegistry',
    'NormRegistry',
    'LRSchedulerRegistry',
    'OptimizerRegistry',
    'RunnerRegistry',
    'CallbackRegistry',
    'LossRegistry',
    'SchedulerRegistry',
    'VisualRegistry',
    'HookRegistry',
    'DistillerRegistry',
    'AccessLayerRegistry',
    'DatasetRegistry',
    'StrategyRegistry',
    'SamplerRegistry',
    'ModelRegistry',
    'TransformRegistry',
    'EnvRegistry',
    'PipelineRegistry',
    'FilterRegistry',
    'ETARegistry',
    'ClipGradRegistry',
    'InitRegistry',
    'CollateRegistry',
]

import inspect
from collections import UserDict
from functools import partial
from typing import Callable, NoReturn, Protocol, cast, no_type_check

import torch
import torch.utils.data
import torch.utils.data.dataset
import torchvision.transforms as tf
from torch import nn

from ..utils import NonInstantiableMeta, get_rank
from .configs import Config
from .logger import logger


class Item(Protocol):
    __name__: str
    __qualname__: str

    def __call__(self, *args, **kwargs):
        ...


class RegistryMeta(  # type: ignore[misc]
    UserDict[str, Item],
    NonInstantiableMeta,
):
    """Meta class for registries.

    Under the hood, registries are simply dictionaries:

        >>> class Cat(metaclass=RegistryMeta): pass
        >>> class BritishShorthair: pass
        >>> Cat['british shorthair'] = BritishShorthair
        >>> Cat['british shorthair']
        <class '...BritishShorthair'>

    Users can also access registries via higher level APIs, i.e. `register_`
    and `build`, for convenience.

    Registries can be subclassed.
    Derived classes of a registry are child registries:

        >>> class HairlessCat(Cat): pass
        >>> Cat.child('HairlessCat')
        <HairlessCat >
    """

    def __init__(cls, *args, **kwargs) -> None:
        """Initialize."""
        UserDict.__init__(cls)
        NonInstantiableMeta.__init__(cls, *args, **kwargs)

    def __missing__(cls, key: str) -> NoReturn:
        """Missing key.

        Args:
            key: the missing key.

        Raises:
            KeyError: always.
        """
        logger.error("%s does not exist in %s", key, cls.__name__)
        raise KeyError(key)

    def __getitem__(cls, key: str) -> Item:
        _, type_ = cls.parse(key)
        return type_

    def __setitem__(cls, key: str, item: Item) -> None:
        """Register ``item`` with name ``key``.

        Args:
            key: name to be registered as.
            item: object to be registered.

        Raises:
            KeyError: if ``key`` is already registered.

        `RegistryMeta` refuses to alter the registered object, in order to
        prevent unintended name clashes:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> Cat['british shorthair'] = 'british shorthair'
            >>> Cat['british shorthair'] = 'BritishShorthair'
            Traceback (most recent call last):
                ...
            KeyError: 'british shorthair'
        """
        if key in cls:  # noqa: E501 pylint: disable=unsupported-membership-test
            logger.error("%s already exist in %s", key, cls)
            raise KeyError(key)
        child, key = cls._parse(key)
        super(RegistryMeta, child).__setitem__(key, item)

    def __delitem__(cls, key: str) -> None:
        child, key = cls._parse(key)
        return super(RegistryMeta, child).__delitem__(key)

    def __contains__(cls, key) -> bool:
        if not isinstance(key, str):
            return False
        child, key = cls._parse(key)
        return super(RegistryMeta, child).__contains__(key)

    def __repr__(cls) -> str:
        items = ' '.join(f'{k}={v}' for k, v in cls.items())
        return f"<{cls.__name__} {items}>"

    @no_type_check
    def __subclasses__(cls=...):
        """Refer to `ABC subclassed by meta classes`_.

        Returns:
            Children registries.

        .. _ABC subclassed by meta classes:
           https://blog.csdn.net/LutingWang/article/details/128320057
        """
        if cls is ...:
            return NonInstantiableMeta.__subclasses__(RegistryMeta)
        return super().__subclasses__()

    def _parse(cls, key: str) -> tuple['RegistryMeta', str]:
        """Parse ``key``.

        Returns:
            The child registry and the updated key.
        """
        if '.' not in key:
            return cls, key
        child_name, key = key.rsplit('.', 1)
        child = cls.child(child_name)
        return child, key

    def child(cls, key: str) -> 'RegistryMeta':
        """Get a direct or indirect derived child registry.

        Args:
            key: dot separated subclass names.

        Raises:
            ValueError: if zero or multiple children are found.

        Returns:
            The derived registry.
        """
        child = cls
        for child_name in key.split('.'):
            subclasses = tuple(child.__subclasses__())  # type: ignore[misc]
            subclasses = tuple(
                subclass for subclass in subclasses
                if subclass.__name__ == child_name
            )
            if len(subclasses) == 0:
                raise ValueError(f"{key} is not a child of {cls}")
            if len(subclasses) > 1:
                raise ValueError(f"{key} matches multiple children of {cls}")
            child, = subclasses
        return child

    def parse(cls, key: str) -> tuple['RegistryMeta', Item]:
        """Parse ``key``.

        Returns:
            The child registry and the corresponding type.
        """
        child, key = cls._parse(key)
        item = super(RegistryMeta, child).__getitem__(key)
        return child, item

    def register_(
        cls,
        *args: str,
        forced: bool = False,
    ) -> Callable[[Item], Item]:
        """Register decorator.

        Args:
            args: names to be registered as.
            forced: if set, registration will always happen.

        Returns:
            Wrapper function.

        `register_` is designed to be an decorator for classes and functions:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> @Cat.register_()
            ... class Munchkin: pass
            >>> @Cat.register_()
            ... def munchkin() -> str:
            ...     return 'munchkin'

        `register_` has the following advantages:

        - default name

          By default, `register_` uses the name of the registered object as
          ``keys``:

          >>> Cat['Munchkin']
          <class '...Munchkin'>
          >>> Cat['munchkin']
          <function munchkin at ...>

        - multiple names

          >>> @Cat.register_('British Longhair', 'british longhair')
          ... class BritishLonghair: pass
          >>> 'British Longhair' in Cat
          True
          >>> 'british longhair' in Cat
          True

        - compatibility with child registries

          >>> class HairlessCat(Cat): pass
          >>> @Cat.register_('HairlessCat.CanadianHairless')
          ... def canadian_hairless() -> str:
          ...     return 'canadian hairless'
          >>> HairlessCat
          <HairlessCat CanadianHairless=<function canadian_hairless at ...>>

        - forced registration

          >>> class AnotherMunchkin: pass
          >>> Cat.register_('Munchkin')(AnotherMunchkin)
          Traceback (most recent call last):
              ...
          KeyError: 'Munchkin'
          >>> Cat.register_('Munchkin', forced=True)(AnotherMunchkin)
          <class '...AnotherMunchkin'>
          >>> Cat['Munchkin']
          <class '...AnotherMunchkin'>
        """

        def wrapper_func(obj: Item) -> Item:
            keys = [obj.__name__] if len(args) == 0 else args
            for key in keys:
                if forced:
                    cls.pop(key, None)
                cls[key] = obj  # noqa: E501 pylint: disable=unsupported-assignment-operation
            return obj

        return wrapper_func

    def _build(cls, type_: Item, config: Config):
        """Build an instance according to the given config.

        Args:
            type_: instance type.
            config: instance specification.

        Returns:
            The built instance.

        To customize the build process of instances, registries must overload
        `_build` with a class method:

            >>> class Cat(metaclass=RegistryMeta):
            ...     @classmethod
            ...     def _build(cls, type_: Item, config: Config):
            ...         obj = RegistryMeta._build(cls, type_, config)
            ...         obj.name = obj.name.upper()
            ...         return obj
            >>> @Cat.register_()
            ... class Munchkin:
            ...     def __init__(self, name: str) -> None:
            ...         self.name = name
            >>> config = Config(type='Munchkin', name='Garfield')
            >>> cat = Cat.build(config)
            >>> cat.name
            'GARFIELD'
        """
        return type_(**config)

    def build(cls, config: Config, **kwargs):
        """Call the registered object to construct a new instance.

        Args:
            config: build parameters.
            kwargs: default configuration.

        Returns:
            The built instance.

        If the registered object is callable, the `build` method will
        automatically call the objects:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> @Cat.register_()
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
        backup_config = default_config.copy()
        config = default_config.copy()

        config_type = config.pop('type')
        registry, type_ = cls.parse(config_type)

        try:
            return registry._build(type_, config)
        except Exception as e:
            # config may be altered
            logger.error("Failed to build\n%s", backup_config.dumps())
            raise e


class PartialRegistryMeta(RegistryMeta):

    @no_type_check
    def __subclasses__(cls=...):
        if cls is ...:
            return NonInstantiableMeta.__subclasses__(PartialRegistryMeta)
        return super().__subclasses__()

    def _build(cls, type_: Item, config: Config):
        return partial(type_, **config)


class Registry(metaclass=RegistryMeta):
    """Base registry.

    To create custom registry, inherit from the `Registry` class:

        >>> class CatRegistry(Registry): pass
    """


class PartialRegistry(metaclass=PartialRegistryMeta):
    pass


class NormRegistry(Registry):
    pass


class OptimizerRegistry(Registry):

    @staticmethod
    def params(model: nn.Module, config: Config) -> Config:
        config = config.copy()
        params = config.pop('params')
        filter_ = FilterRegistry.build(params)
        filtered_params = [p for _, p in filter_(model)]
        assert all(p.requires_grad for p in filtered_params)
        config.params = filtered_params
        return config

    @classmethod
    def _build(cls, type_: Item, config: Config):
        model: nn.Module = config.pop('model')
        params = config.pop('params', None)
        if params is None:
            config.params = [p for p in model.parameters() if p.requires_grad]
        else:
            config.params = [cls.params(model, p) for p in params]
        return RegistryMeta._build(cls, type_, config)


class LRSchedulerRegistry(Registry):

    @classmethod
    def _build(cls, type_: Item, config: Config):
        if type_ is torch.optim.lr_scheduler.SequentialLR:
            config.schedulers = [
                cls.build(scheduler, optimizer=config.optimizer)
                for scheduler in config.schedulers
            ]
        return RegistryMeta._build(cls, type_, config)


class RunnerRegistry(Registry):
    pass


class CallbackRegistry(Registry):
    pass


class LossRegistry(Registry):
    pass


class SchedulerRegistry(Registry):

    @classmethod
    def _build(cls, type_: Item, config: Config):
        from ..losses.schedulers import (  # noqa: E501 pylint: disable=import-outside-toplevel
            ChainedScheduler,
            SequentialScheduler,
        )
        if type_ in (SequentialScheduler, ChainedScheduler):
            config.schedulers = [
                cls.build(scheduler) for scheduler in config.schedulers
            ]
        return RegistryMeta._build(cls, type_, config)


class VisualRegistry(Registry):
    pass


class HookRegistry(Registry):
    pass


class DistillerRegistry(Registry):
    pass


class DatasetRegistry(Registry):

    @classmethod
    def _build(cls, type_: Item, config: Config):
        if type_ is torch.utils.data.ConcatDataset:
            config.datasets = list(map(cls.build, config.datasets))
        return RegistryMeta._build(cls, type_, config)


class AccessLayerRegistry(Registry):
    pass


class StrategyRegistry(Registry):
    pass


class SamplerRegistry(Registry):
    pass


class ModelRegistry(Registry):

    @classmethod
    def init_weights(
        cls,
        model: nn.Module,
        config: Config | None,
        prefix: str = '',
    ) -> None:
        weights = f"{model.__class__.__name__} ({prefix}) weights"

        if getattr(model, '__initialized', False):
            if get_rank() == 0:
                logger.debug("Skip re-initializing %s", weights)
            return
        setattr(model, '__initialized', True)  # noqa: B010

        if config is None:
            if get_rank() == 0:
                logger.debug(
                    "Skip initializing %s since config is None",
                    weights,
                )
            return

        init_weights: Callable[[Config], bool] | None = \
            getattr(model, 'init_weights', None)
        if init_weights is not None:
            if get_rank() == 0:
                logger.debug("Initializing %s with %s", weights, config)
            recursive = init_weights(config)
            if not recursive:
                return

        for (
            name,  # noqa: E501 pylint: disable=redefined-outer-name
            child,
        ) in model.named_children():
            cls.init_weights(child, config, f'{prefix}.{name}')

    @classmethod
    def _build(cls, type_: Item, config: Config):
        config = config.copy()
        init_weights = config.pop('init_weights', Config())
        model = RegistryMeta._build(cls, type_, config)
        if isinstance(model, nn.Module):
            cls.init_weights(model, init_weights)
        return model


class TransformRegistry(Registry):

    @classmethod
    def _build(cls, type_: Item, config: Config):
        if type_ is tf.Compose:
            config.transforms = list(map(cls.build, config.transforms))
        return RegistryMeta._build(cls, type_, config)


class EnvRegistry(Registry):
    pass


class PipelineRegistry(Registry):
    pass


class FilterRegistry(Registry):
    pass


class ETARegistry(Registry):
    pass


class ClipGradRegistry(PartialRegistry):
    pass


class InitRegistry(PartialRegistry):
    pass


class CollateRegistry(PartialRegistry):
    pass


def descendant_classes(cls: type) -> list[type]:
    classes = []
    for subclass in cls.__subclasses__():
        classes.append(subclass)
        classes.extend(descendant_classes(subclass))
    return classes


c: type

for c in descendant_classes(nn.Module):
    # pylint: disable=invalid-name
    name = '_'.join(c.__module__.split('.') + [c.__name__])
    if name not in ModelRegistry:
        ModelRegistry.register_(name)(cast(Item, c))

for _, c in inspect.getmembers(torch.utils.data.dataset, inspect.isclass):
    if issubclass(c, torch.utils.data.Dataset):
        DatasetRegistry.register_()(cast(Item, c))

for c in descendant_classes(torch.utils.data.Sampler):
    SamplerRegistry.register_()(cast(Item, c))

for c in descendant_classes(torch.optim.Optimizer):
    if '<locals>' not in c.__qualname__:
        OptimizerRegistry.register_()(cast(Item, c))

for c in descendant_classes(torch.optim.lr_scheduler.LRScheduler):
    LRSchedulerRegistry.register_()(cast(Item, c))

NormRegistry.update(
    BN1d=nn.BatchNorm1d,
    BN2d=nn.BatchNorm2d,
    BN=nn.BatchNorm2d,
    BN3d=nn.BatchNorm3d,
    SyncBN=nn.SyncBatchNorm,
    GN=nn.GroupNorm,
    LN=nn.LayerNorm,
    IN1d=nn.InstanceNorm1d,
    IN2d=nn.InstanceNorm2d,
    IN=nn.InstanceNorm2d,
    IN3d=nn.InstanceNorm3d,
)

for _, c in inspect.getmembers(tf, inspect.isclass):
    TransformRegistry.register_()(cast(Item, c))

ClipGradRegistry.register_()(nn.utils.clip_grad_norm_)
ClipGradRegistry.register_()(nn.utils.clip_grad_value_)

InitRegistry.register_()(nn.init.uniform_)
InitRegistry.register_()(nn.init.normal_)
InitRegistry.register_()(nn.init.trunc_normal_)
InitRegistry.register_()(nn.init.constant_)
InitRegistry.register_()(nn.init.ones_)
InitRegistry.register_()(nn.init.zeros_)
InitRegistry.register_()(nn.init.eye_)
InitRegistry.register_()(nn.init.dirac_)
InitRegistry.register_()(nn.init.xavier_uniform_)
InitRegistry.register_()(nn.init.xavier_normal_)
InitRegistry.register_()(nn.init.kaiming_uniform_)
InitRegistry.register_()(nn.init.kaiming_normal_)
InitRegistry.register_()(nn.init.orthogonal_)
InitRegistry.register_()(nn.init.sparse_)
