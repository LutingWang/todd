# pylint: disable=no-value-for-parameter

__all__ = [
    'RegistryMeta',
    'PartialRegistryMeta',
    'Registry',
    'PartialRegistry',
    'NormRegistry',
    'LrSchedulerRegistry',
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
    'ClipGradRegistry',
    'InitRegistry',
    'CollateRegistry',
]

import inspect
import re
from collections import UserDict
from functools import partial
from typing import Any, Callable, NoReturn, TypeVar, no_type_check

import torch
import torch.utils.data
import torchvision.transforms as tf
from torch import nn

from .. import utils
from ..utils import NonInstantiableMeta, get_rank
from .configs import Config
from .logger import logger

T = TypeVar('T', bound=Callable)


class RegistryMeta(  # type: ignore[misc]
    UserDict[str, Any],
    NonInstantiableMeta,
):
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

    def __init__(cls, *args, **kwargs) -> None:
        """Initialize."""
        UserDict.__init__(cls)
        NonInstantiableMeta.__init__(cls, *args, **kwargs)

    def __call__(cls, *args, **kwargs) -> NoReturn:
        """Prevent `Registry` classes from being initialized.

        Raises:
            TypeError: always.
        """
        raise TypeError("Registry cannot be initialized")

    def __missing__(cls, key: str) -> NoReturn:
        """Missing key.

        Args:
            key: the missing key.

        Raises:
            KeyError: always.
        """
        logger.error("%s does not exist in %s", key, cls.__name__)
        raise KeyError(key)

    def __repr__(cls) -> str:
        items = ' '.join(f'{k}={v}' for k, v in cls.items())
        return f"<{cls.__name__} {items}>"

    def __setitem__(cls, key: str, item, forced: bool = False) -> None:
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
        if not forced and key in cls:
            logger.error("%s already exist in %s", key, cls.__name__)
            raise KeyError(key)
        return super().__setitem__(key, item)

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

    def child(cls, key: str) -> 'RegistryMeta':
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
                for subclass in cls.__subclasses__()  # type: ignore[misc]
                if subclass.__name__ == child_name
            )
            if len(subclasses) == 0:
                raise ValueError(f"{key} is not a child of {cls}")
            if len(subclasses) > 1:
                raise ValueError(f"{key} matches multiple children of {cls}")
            cls, = subclasses  # pylint: disable=self-cls-assignment
        return cls

    def _parse(cls, key: str) -> tuple['RegistryMeta', str]:
        """Parse the child name from the ``key``.

        Returns:
            The child registry and the name to be registered.
        """
        if '.' not in key:
            return cls, key
        child_name, key = key.rsplit('.', 1)
        return cls.child(child_name), key

    def register_(
        cls,
        *args: str,
        **kwargs,
    ) -> Callable[[T], T]:
        """Register decorator.

        Args:
            args: names to be registered as.
            kwargs: refer to `__setitem__`.

        Returns:
            Wrapper function.

        `register` is designed to be an decorator for classes and functions:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> @Cat.register_()
            ... class Munchkin: pass
            >>> @Cat.register_()
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
        """

        def wrapper_func(obj: T) -> T:
            keys = [obj.__name__] if len(args) == 0 else args
            for key in keys:
                registry, key = cls._parse(key)
                registry.__setitem__(  # noqa: E501 pylint: disable=unnecessary-dunder-call
                    key, obj, **kwargs,)
            return obj

        return wrapper_func

    def _build(cls, config: Config):
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
            >>> @Cat.register_()
            ... class Munchkin: pass
            >>> Cat.build(Config(type='Munchkin'))
            'MUNCHKIN'
        """
        type_ = cls[config.pop('type')]
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

        config = default_config.copy()
        registry, config.type = cls._parse(config.type)

        try:
            return registry._build(config)
        except Exception as e:
            # config may be altered
            logger.error("Failed to build\n%s", default_config.dumps())
            raise e


class PartialRegistryMeta(RegistryMeta):

    @no_type_check
    def __subclasses__(cls=...):
        if cls is ...:
            return NonInstantiableMeta.__subclasses__(PartialRegistryMeta)
        return super().__subclasses__()

    def _build(cls, config: Config) -> partial:
        type_ = cls[config.pop('type')]
        if inspect.isclass(type_):
            return type_(**config)
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
        # TODO: add more matchers except re
        config = config.copy()
        config.params = [
            p for n, p in model.named_parameters()
            if re.match(config.params, n) and p.requires_grad
        ]
        return config

    @classmethod
    def _build(cls, config: Config) -> torch.optim.Optimizer:
        model: nn.Module = config.pop('model')
        params = config.pop('params', None)
        if params is None:
            config.params = [p for p in model.parameters() if p.requires_grad]
        else:
            config.params = [cls.params(model, p) for p in params]
        return RegistryMeta._build(cls, config)


class LrSchedulerRegistry(Registry):  # TODO: rename to LR...

    @classmethod
    def _build(cls, config: Config) -> torch.optim.lr_scheduler.LRScheduler:
        if config.type == torch.optim.lr_scheduler.SequentialLR.__name__:
            config.schedulers = [
                cls.build(scheduler, optimizer=config.optimizer)
                for scheduler in config.schedulers
            ]
        return RegistryMeta._build(cls, config)


class RunnerRegistry(Registry):
    pass


class CallbackRegistry(Registry):
    pass


class LossRegistry(Registry):
    pass


class SchedulerRegistry(Registry):

    @classmethod
    def _build(cls, config: Config) -> torch.optim.lr_scheduler.LRScheduler:
        from ..losses.schedulers import ChainedScheduler, SequentialScheduler
        if config.type in (
            SequentialScheduler.__name__,
            ChainedScheduler.__name__,
        ):
            config.schedulers = [
                cls.build(scheduler) for scheduler in config.schedulers
            ]
        return RegistryMeta._build(cls, config)


class VisualRegistry(Registry):
    pass


class HookRegistry(Registry):
    pass


class DistillerRegistry(Registry):
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
        setattr(model, '__initialized', True)

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

        for name, child in model.named_children():  # noqa: E501 pylint: disable=redefined-outer-name
            cls.init_weights(child, config, f'{prefix}.{name}')

    @classmethod
    def _build(cls, config: Config):
        config = config.copy()
        init_weights = config.pop('init_weights', Config())
        model = RegistryMeta._build(cls, config)
        if isinstance(model, nn.Module):
            cls.init_weights(model, init_weights)
        return model


class TransformRegistry(Registry):

    @classmethod
    def _build(cls, config: Config):
        if config.type == tf.Compose.__name__:
            config.transforms = list(map(cls.build, config.transforms))
        return RegistryMeta._build(cls, config)


class EnvRegistry(Registry):
    pass


class PipelineRegistry(Registry):
    pass


class ClipGradRegistry(PartialRegistry):
    pass


class InitRegistry(PartialRegistry):
    pass


class CollateRegistry(PartialRegistry):
    pass


c: type

for c in torch.nn.Module.__subclasses__():
    module = c.__module__.replace('.', '_')
    name = c.__name__
    ModelRegistry.register_(f'{module}_{name}')(c)

for c in torch.utils.data.Sampler.__subclasses__():
    SamplerRegistry.register_()(c)

for c in torch.optim.Optimizer.__subclasses__():
    OptimizerRegistry.register_()(c)

for c in torch.optim.lr_scheduler.LRScheduler.__subclasses__():
    LrSchedulerRegistry.register_()(c)

NormRegistry['BN1d'] = nn.BatchNorm1d
NormRegistry['BN2d'] = NormRegistry['BN'] = nn.BatchNorm2d
NormRegistry['BN3d'] = nn.BatchNorm3d
NormRegistry['SyncBN'] = nn.SyncBatchNorm
NormRegistry['GN'] = nn.GroupNorm
NormRegistry['LN'] = nn.LayerNorm
NormRegistry['IN1d'] = nn.InstanceNorm1d
NormRegistry['IN2d'] = NormRegistry['IN'] = nn.InstanceNorm2d
NormRegistry['IN3d'] = nn.InstanceNorm3d

for _, c in inspect.getmembers(tf, inspect.isclass):
    TransformRegistry.register_()(c)

EnvRegistry.register_('Platform')(utils.platform)
EnvRegistry.register_('NVIDIA SMI')(utils.nvidia_smi)
EnvRegistry.register_('Python version')(utils.python_version)
EnvRegistry.register_('PyTorch version')(utils.pytorch_version)
EnvRegistry.register_('TorchVision version')(utils.torchvision_version)
EnvRegistry.register_('OpenCV version')(utils.opencv_version)
EnvRegistry.register_('Todd version')(utils.todd_version)
EnvRegistry.register_('CUDA_HOME')(utils.cuda_home)
EnvRegistry.register_('Git commit ID')(utils.git_commit_id)
EnvRegistry.register_('Git status')(utils.git_status)

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
