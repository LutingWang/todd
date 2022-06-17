from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Type,
    TypeVar,
    overload,
    runtime_checkable,
)

import torch.nn as nn

__all__ = [
    'NORM_LAYERS',
    'Registry',
]


@runtime_checkable
class ModuleProto(Protocol):
    __name__: str


ModuleType = TypeVar('ModuleType', bound=ModuleProto)
BuildFunc = Callable[['Registry', Dict[str, Any]], Any]


class Registry:
    _build_func: Optional[BuildFunc]

    def __init__(
        self,
        name: str,
        base: Optional[Type[ModuleType]] = None,
        parent: Optional['Registry'] = None,
        build_func: Optional[BuildFunc] = None,
    ) -> None:
        assert '.' not in name
        if parent is not None:
            parent._children[name] = (self)
            if build_func is None:
                build_func = parent._build_func

        self._name = name
        self._base = base
        self._parent = parent
        self._build_func = build_func

        self._modules: Dict[str, ModuleProto] = dict()
        self._children: Dict[str, 'Registry'] = dict()

    @property
    def name(self) -> str:
        return self._name

    @property
    def parent(self) -> Optional['Registry']:
        return self._parent

    @property
    def modules(self) -> Dict[str, ModuleProto]:
        return self._modules

    @property
    def children(self) -> Dict[str, 'Registry']:
        return self._children

    @property
    def root(self) -> 'Registry':
        if self._parent is None:
            return self
        return self._parent.root

    def __len__(self) -> int:
        return len(self._modules)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: str) -> ModuleProto:
        item = self.get(key)
        if item is None:
            raise KeyError(f'{key} does not exist in {self.name} registry')
        return item

    def _register_module(
        self,
        cls: ModuleType,
        name: Optional[str] = None,
        aliases: Iterable[str] = tuple(),
        force: bool = False,
    ) -> ModuleType:
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
    ) -> Callable[[ModuleType], ModuleType]:
        ...

    @overload
    def register_module(
        self,
        cls: ModuleType,
        *,
        name: Optional[str] = None,
        aliases: Iterable[str] = tuple(),
        force: bool = False,
    ) -> ModuleType:
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

    def get(self, key: str) -> Optional[ModuleProto]:
        if '.' not in key:
            return self._modules.get(key)
        name, subkey = key.split('.', 1)
        if name not in self._children:
            return None
        return self._children[name].get(subkey)

    @overload
    def build(
        self,
        cfg: ModuleType,
    ) -> ModuleType:
        pass

    @overload
    def build(
        self,
        cfg: dict,
        default_args: Optional[dict] = None,
    ) -> ModuleType:
        pass

    def build(self, cfg, default_args=None):
        if self._base is not None and isinstance(cfg, self._base):
            if default_args is not None:
                raise ValueError(
                    '`default_args` is not supported when `cfg` is an '
                    f'instance of {self._base.__name__}'
                )
            return cfg

        if not isinstance(cfg, dict):
            if self._base is None:
                message = f"cfg must be a dictionary, but got {type(cfg)}"
            else:
                message = (
                    f"cfg must be an instance of {self._base.__name__} or a "
                    f"dictionary, but got {type(cfg)}"
                )
            raise TypeError(message)
        cfg = cfg.copy()
        if default_args is not None:
            if not isinstance(default_args, dict):
                raise TypeError(
                    "default_args must be a dictionary, but got "
                    f"{type(default_args)}"
                )
            for k, v in default_args.items():
                cfg.setdefault(k, v)

        if self._build_func is not None:
            return self._build_func(self, cfg)

        if 'type' not in cfg:
            raise KeyError(f'{cfg} cfg does not specify type')

        type_ = cfg.pop('type')
        if isinstance(type_, str):
            type_ = self[type_]
        try:
            return type_(**cfg)
        except Exception as e:
            # raise type(e)(f'{type_.__name__}: {e}')
            raise RuntimeError(
                f'Registry {self.name} failed to build {type_.__name__} with '
                f'cfg {cfg}. {e.__class__.__name__}: {e}'
            )


NORM_LAYERS = Registry('norm layers')
NORM_LAYERS.register_module(cls=nn.BatchNorm1d, name='BN1d')
NORM_LAYERS.register_module(cls=nn.BatchNorm2d, name='BN2d', aliases=['BN'])
NORM_LAYERS.register_module(cls=nn.BatchNorm3d, name='BN3d')
NORM_LAYERS.register_module(cls=nn.SyncBatchNorm, name='SyncBN')
NORM_LAYERS.register_module(cls=nn.GroupNorm, name='GN')
NORM_LAYERS.register_module(cls=nn.LayerNorm, name='LN')
NORM_LAYERS.register_module(cls=nn.InstanceNorm1d, name='IN1d')
NORM_LAYERS.register_module(cls=nn.InstanceNorm2d, name='IN2d', aliases=['IN'])
NORM_LAYERS.register_module(cls=nn.InstanceNorm3d, name='IN3d')
