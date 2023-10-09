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
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as tf

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

    @no_type_check
    def __subclasses__(self=...):
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

          >>> @Cat.register('British Longhair', 'british longhair')
          ... class BritishLonghair: pass
          >>> 'British Longhair' in Cat
          True
          >>> 'british longhair' in Cat
          True

        - compatibility with child registries

          >>> class HairlessCat(Cat): pass
          >>> @Cat.register('HairlessCat.CanadianHairless')
          ... def canadian_hairless() -> str:
          ...     return 'canadian hairless'
          >>> HairlessCat
          <HairlessCat CanadianHairless=<function canadian_hairless at ...>>
        """

        def wrapper_func(obj: T) -> T:
            keys = [obj.__name__] if len(args) == 0 else args
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


class PartialRegistryMeta(RegistryMeta):

    @no_type_check
    def __subclasses__(self=...):
        if self is ...:
            return NonInstantiableMeta.__subclasses__(PartialRegistryMeta)
        return super().__subclasses__()

    def _build(self, config: Config) -> partial:
        type_ = self[config.pop('type')]
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
                logger.debug(f"Skip re-initializing {weights}")
            return
        setattr(model, '__initialized', True)

        if config is None:
            if get_rank() == 0:
                logger.debug(
                    f"Skip initializing {weights} since config is None"
                )
            return

        init_weights: Callable[[Config], bool] | None = \
            getattr(model, 'init_weights', None)
        if init_weights is not None:
            if get_rank() == 0:
                logger.debug(f"Initializing {weights} with {config}")
            recursive = init_weights(config)
            if not recursive:
                return

        for name, child in model.named_children():
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
    ModelRegistry.register(f'{module}_{name}')(c)

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

for _, c in inspect.getmembers(tf, inspect.isclass):
    TransformRegistry.register()(c)

EnvRegistry.register('Platform')(utils.platform)
EnvRegistry.register('NVIDIA SMI')(utils.nvidia_smi)
EnvRegistry.register('Python version')(utils.python_version)
EnvRegistry.register('PyTorch version')(utils.pytorch_version)
EnvRegistry.register('TorchVision version')(utils.torchvision_version)
EnvRegistry.register('OpenCV version')(utils.opencv_version)
EnvRegistry.register('Todd version')(utils.todd_version)
EnvRegistry.register('CUDA_HOME')(utils.cuda_home)
EnvRegistry.register('Git commit ID')(utils.git_commit_id)
EnvRegistry.register('Git status')(utils.git_status)

ClipGradRegistry.register()(nn.utils.clip_grad_norm_)
ClipGradRegistry.register()(nn.utils.clip_grad_value_)

InitRegistry.register()(nn.init.uniform_)
InitRegistry.register()(nn.init.normal_)
InitRegistry.register()(nn.init.trunc_normal_)
InitRegistry.register()(nn.init.constant_)
InitRegistry.register()(nn.init.ones_)
InitRegistry.register()(nn.init.zeros_)
InitRegistry.register()(nn.init.eye_)
InitRegistry.register()(nn.init.dirac_)
InitRegistry.register()(nn.init.xavier_uniform_)
InitRegistry.register()(nn.init.xavier_normal_)
InitRegistry.register()(nn.init.kaiming_uniform_)
InitRegistry.register()(nn.init.kaiming_normal_)
InitRegistry.register()(nn.init.orthogonal_)
InitRegistry.register()(nn.init.sparse_)
