import inspect
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import torch.nn as nn

from ._extensions import get_logger

__all__ = [
    'NORM_LAYERS',
    'Registry',
]

T = TypeVar('T')


class _Registry(Generic[T]):

    def __init__(self, name: str) -> None:
        assert '.' not in name
        self._name = name

        self._modules: Dict[str, Type] = dict()
        self._logger = get_logger()

    @property
    def name(self) -> str:
        return self._name

    @property
    def modules(self) -> Dict[str, Type[T]]:
        return self._modules

    def __len__(self) -> int:
        return len(self._modules)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: str) -> Type[T]:
        item: Optional[Type[T]] = self.get(key)
        if item is None:
            raise KeyError(f'{key} does not exist in {self.name} registry')
        return item

    def _register_module(
        self,
        cls: Type[T],
        name: Optional[str] = None,
        aliases: Iterable[str] = tuple(),
        force: bool = False,
    ) -> Type[T]:
        if inspect.isabstract(cls):
            raise TypeError(f'{cls} is an abstract class')
        if name is None:
            name = cls.__name__
        names = [name] + list(aliases)
        if not force and len(names & self._modules.keys()) > 0:
            raise KeyError(f'{name} already exists in {self.name} registry')
        self._modules.update({name: cls for name in names})
        return cls

    @overload
    def register_module(
        self,
        *,
        name: Optional[str] = None,
        aliases: Iterable[str] = tuple(),
        force: bool = False,
    ) -> Callable[[Type[T]], Type[T]]:
        ...

    @overload
    def register_module(
        self,
        cls: Type[T],
        *,
        name: Optional[str] = None,
        aliases: Iterable[str] = tuple(),
        force: bool = False,
    ) -> Type[T]:
        ...

    def register_module(
        self,
        cls=None,
        *,
        name=None,
        aliases=tuple(),
        force=False,
    ):
        register = partial(
            self._register_module,
            name=name,
            aliases=aliases,
            force=force,
        )
        if cls is None:
            return register
        return register(cls=cls)

    def get(self, key: str) -> Optional[Type[T]]:
        return self._modules.get(key)

    def _build(self, cfg: dict) -> T:
        if 'type' not in cfg:
            raise KeyError(f'{cfg} cfg does not specify type')

        type_ = cfg.pop('type')
        if isinstance(type_, str):
            type_ = self[type_]
        try:
            return type_(**cfg)
        except Exception as e:
            self._logger.error(
                f'Failed to build {type_.__name__} with config {cfg}',
            )
            raise e

    def build(
        self,
        cfg: dict,
        default_args: Optional[dict] = None,
    ) -> T:
        cfg = cfg.copy()
        if default_args is not None:
            if not isinstance(default_args, dict):
                raise TypeError(
                    "default_args must be a dictionary, but got "
                    f"{type(default_args)}"
                )
            for k, v in default_args.items():
                cfg.setdefault(k, v)
        return self._build(cfg)


class _ParentMixin(_Registry[T]):

    def __init__(
        self,
        *args,
        parent: Optional['Registry'] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._children: Dict[str, 'Registry'] = dict()

        if parent is None:
            return
        parent._children[self._name] = cast(Registry, self)
        self._parent = parent

    @property
    def children(self) -> Dict[str, 'Registry']:
        return self._children

    def index_descendent(self, name: str) -> 'Registry':
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

    @property
    def parent(self) -> 'Registry':
        return self._parent

    @property
    def root(self) -> 'Registry':
        if self.has_parent():
            return self._parent.root
        return cast(Registry, self)

    def get(self, key: str) -> Optional[Type[T]]:
        if '.' not in key:
            return super().get(key)
        descendent_name, key = key.rsplit('.', 1)
        descendent = self.get_descendent(descendent_name)
        if descendent is None:
            return None
        return descendent.get(key)


class _BaseMixin(_ParentMixin[T]):

    def __init__(
        self,
        *args,
        base: Optional[Type[T]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        parent_has_base = self.has_parent() and self.parent.has_base()
        if base is None:
            if parent_has_base:
                base = self.parent.base
            else:
                return
        else:
            if parent_has_base and not issubclass(base, self.parent.base):
                raise TypeError(
                    f'{base} is not a subclass of {self.parent.base}',
                )

        self._base = cast(Type[T], base)
        if not inspect.isabstract(base):
            self._register_module(base)

    def has_base(self) -> bool:
        return hasattr(self, '_base')

    @property
    def base(self) -> Type[T]:
        return self._base

    def _register_module(
        self,
        cls: Type[T],
        *args,
        **kwargs,
    ) -> Type[T]:
        if self.has_base() and not issubclass(cls, self._base):
            raise TypeError(f'{cls} is not a subclass of {self._base}')
        return super()._register_module(cls, *args, **kwargs)

    def build(
        self,
        cfg: Union[dict, T],
        default_args: Optional[dict] = None,
    ) -> T:
        if not self.has_base() or not isinstance(cfg, self._base):
            if not isinstance(cfg, dict):
                raise TypeError(
                    'cfg must be a dictionary or a subclass of `self._base`'
                    f'but got {type(cfg)}'
                )
            return super().build(cfg, default_args)

        return cfg


class _BuildFuncMixin(_ParentMixin[T]):

    def __init__(
        self,
        *args,
        build_func: Optional[Callable[['Registry', Dict[str, Any]], T]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if build_func is None:
            if not self.has_parent() or not self.parent.has_build_func():
                return
            build_func = self.parent.build_func
        self._build_func = build_func

    def has_build_func(self) -> bool:
        return hasattr(self, '_build_func')

    @property
    def build_func(self) -> Callable[['Registry', Dict[str, Any]], T]:
        return self._build_func

    def _build(self, cfg: dict) -> T:
        if self.has_build_func():
            return self._build_func(cast(Registry, self), cfg)
        return super()._build(cfg)


class Registry(
    _BuildFuncMixin[T],
    _BaseMixin[T],
    _ParentMixin[T],
    _Registry[T],
):
    pass


NORM_LAYERS: Registry[nn.Module] = Registry('norm layers', base=nn.Module)
NORM_LAYERS.register_module(cls=nn.BatchNorm1d, name='BN1d')
NORM_LAYERS.register_module(cls=nn.BatchNorm2d, name='BN2d', aliases=['BN'])
NORM_LAYERS.register_module(cls=nn.BatchNorm3d, name='BN3d')
NORM_LAYERS.register_module(cls=nn.SyncBatchNorm, name='SyncBN')
NORM_LAYERS.register_module(cls=nn.GroupNorm, name='GN')
NORM_LAYERS.register_module(cls=nn.LayerNorm, name='LN')
NORM_LAYERS.register_module(cls=nn.InstanceNorm1d, name='IN1d')
NORM_LAYERS.register_module(cls=nn.InstanceNorm2d, name='IN2d', aliases=['IN'])
NORM_LAYERS.register_module(cls=nn.InstanceNorm3d, name='IN3d')
