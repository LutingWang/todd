import logging
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch.nn as nn

from ._extensions import Config, get_logger

__all__ = [
    'NORM_LAYERS',
    'Registry',
]

T = TypeVar('T')


class _RegistryProto(Protocol[T]):  # TODO: inherit from MutatbleMapping
    _name: str
    _logger: logging.Logger
    _modules: Dict[str, Type[T]]
    _children: Dict[str, 'Registry']
    _parent: 'Registry'
    _base: Type[T]
    _build_func: Callable[['Registry', Config], T]

    def __getitem__(self, key: str) -> Type[T]:
        pass

    def get(self, key: str) -> Optional[Type[T]]:
        pass

    def register(
        self,
        cls: Type[T],
        *,
        name: Optional[str] = None,
        aliases: Iterable[str] = tuple(),
        force: bool = False,
    ) -> Type[T]:
        pass

    def descendent(self, name: str) -> 'Registry[T]':
        pass

    def get_descendent(self, name: str) -> Optional['Registry[T]']:
        pass

    def has_parent(self) -> bool:
        pass


class _ModulesMixin(_RegistryProto[T]):

    def __init__(self) -> None:
        self._modules: Dict[str, Type[T]] = dict()

    def __len__(self) -> int:
        return len(self._modules)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: str) -> Type[T]:
        item: Optional[Type[T]] = self.get(key)
        if item is None:
            raise KeyError(f'{key} does not exist in {self._name} registry')
        return item

    @property
    def modules(self) -> Dict[str, Type[T]]:
        return self._modules

    def register(
        self,
        cls: Type[T],
        *,
        name: Optional[str] = None,
        aliases: Iterable[str] = tuple(),
        force: bool = False,
    ) -> Type[T]:
        if name is None:
            name = cls.__name__
        names = [name] + list(aliases)
        if not force and len(names & self._modules.keys()) > 0:
            raise KeyError(f'{name} already exists in {self._name} registry')
        self._modules.update({name: cls for name in names})
        return cls

    def register_module(
        self,
        *,
        name: Optional[str] = None,
        aliases: Iterable[str] = tuple(),
        force: bool = False,
    ) -> Callable[[Type[T]], Type[T]]:
        return partial(
            self.register,
            name=name,
            aliases=aliases,
            force=force,
        )

    def get(self, key: str) -> Optional[Type[T]]:
        return self._modules.get(key)

    def build(self, config: Config) -> T:
        config = config.copy()
        type_ = config.pop('type')
        if isinstance(type_, str):
            type_ = self[type_]
        try:
            return type_(**config)
        except Exception as e:
            self._logger.error(
                f'Failed to build {type_.__name__} with config {config}',
            )
            raise e


class _ParentMixin(_RegistryProto[T]):

    def __init__(
        self,
        parent: Optional['Registry'] = None,
    ) -> None:
        self._children: Dict[str, 'Registry'] = dict()

        if parent is not None:
            parent._children[self._name] = cast(Registry, self)
            self._parent = parent

    @property
    def children(self) -> Dict[str, 'Registry']:
        return self._children

    @property
    def parent(self) -> 'Registry':
        return self._parent

    @property
    def root(self) -> 'Registry':
        if self.has_parent():
            return self._parent.root
        return cast(Registry, self)

    def descendent(self, name: str) -> 'Registry':
        descendent = self.get_descendent(name)
        if descendent is None:
            raise KeyError(
                f"Registry {self._name} does not have descendent named {name}."
            )
        return descendent

    def get_descendent(self, name: str) -> Optional['Registry']:
        current = self
        for child_name in name.split('.'):
            if child_name not in current._children:
                return None
            current = current._children[child_name]
        return cast(Registry, current)

    def has_parent(self) -> bool:
        return hasattr(self, '_parent')


class _BaseMixin(_RegistryProto[T]):

    def __init__(
        self,
        base: Optional[Type[T]] = None,
        register_base: bool = False,
    ) -> None:
        if base is None:
            return
        self._base = base
        if register_base:
            self.register(base)

    @property
    def base(self) -> Type[T]:
        return self._base

    def has_base(self) -> bool:
        return hasattr(self, '_base')


class _BuildFuncMixin(_RegistryProto[T]):

    def __init__(
        self,
        build_func: Optional[Callable[['Registry', Config], T]] = None,
    ) -> None:
        if build_func is not None:
            self._build_func = build_func

    @property
    def build_func(self) -> Callable[['Registry', Config], T]:
        return self._build_func

    def has_build_func(self) -> bool:
        return hasattr(self, '_build_func')


class Registry(
    _ModulesMixin[T],
    _ParentMixin[T],
    _BaseMixin[T],
    _BuildFuncMixin[T],
):

    def __init__(
        self,
        name: str,
        *,
        parent: Optional['Registry'] = None,
        base: Optional[Type[T]] = None,
        register_base: bool = False,
        build_func: Optional[Callable[['Registry', Config], T]] = None,
    ) -> None:
        assert '.' not in name
        self._name = name
        self._logger = get_logger()

        _ModulesMixin.__init__(self)
        _ParentMixin.__init__(self, parent)

        if (
            base is not None and self.has_parent() and self._parent.has_base()
            and not issubclass(base, self._parent._base)
        ):
            raise TypeError(
                f'{base} is not a subclass of {self._parent._base}',
            )
        if (base is None and self.has_parent() and self._parent.has_base()):
            base = self._parent._base
        _BaseMixin.__init__(self, base, register_base)

        if (
            build_func is None and self.has_parent()
            and self._parent.has_build_func()
        ):
            build_func = self._parent._build_func
        _BuildFuncMixin.__init__(self, build_func)

    @property
    def name(self) -> str:
        return self._name

    def register(
        self,
        cls: Type[T],
        *args,
        **kwargs,
    ) -> Type[T]:
        if self.has_base() and not issubclass(cls, self._base):
            raise TypeError(f'{cls} is not a subclass of {self._base}')
        return _ModulesMixin.register(self, cls, *args, **kwargs)

    def get(self, key: str) -> Optional[Type[T]]:
        if '.' not in key:
            return _ModulesMixin.get(self, key)
        descendent_name, key = key.rsplit('.', 1)
        descendent = self.get_descendent(descendent_name)
        if descendent is None:
            return None
        return descendent.get(key)

    def build(
        self,
        config: Union[Config, T],
        default_args: Optional[Config] = None,
    ) -> T:
        if self.has_base():
            if isinstance(config, self._base):
                return config
            if not isinstance(config, dict):
                raise TypeError(
                    f"{config.__class__.__name__} is neither dict nor "
                    f"{self._base.__name__}"
                )
        else:
            if not isinstance(config, dict):
                return config
        default_args = dict() if default_args is None else default_args.copy()
        default_args.update(config)
        if self.has_build_func():
            return self._build_func(self, default_args)
        return _ModulesMixin.build(self, default_args)


NORM_LAYERS: Registry[nn.Module] = Registry('norm layers', base=nn.Module)
NORM_LAYERS.register(nn.BatchNorm1d, name='BN1d')
NORM_LAYERS.register(nn.BatchNorm2d, name='BN2d', aliases=['BN'])
NORM_LAYERS.register(nn.BatchNorm3d, name='BN3d')
NORM_LAYERS.register(nn.SyncBatchNorm, name='SyncBN')
NORM_LAYERS.register(nn.GroupNorm, name='GN')
NORM_LAYERS.register(nn.LayerNorm, name='LN')
NORM_LAYERS.register(nn.InstanceNorm1d, name='IN1d')
NORM_LAYERS.register(nn.InstanceNorm2d, name='IN2d', aliases=['IN'])
NORM_LAYERS.register(nn.InstanceNorm3d, name='IN3d')
