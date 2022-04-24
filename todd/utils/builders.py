from typing import Any, Callable, Tuple, Type, TypeVar, Union

from mmcv.runner import BaseModule
from mmcv.utils import Registry


T = TypeVar('T')


def build(cls, cfg, **kwargs) -> BaseModule:
    if cfg is None: return None
    module = cfg if isinstance(cfg, cls) else cls(cfg, **kwargs)
    return module


def build_metas(name: str, base: Type[T]) -> Tuple[Registry, Type[Union[T, dict]], Callable[[Union[T, dict]], T]]:
    registry = Registry(name)
    config_type = Union[base, dict]

    def build_func(cfg: config_type) -> base:
        if isinstance(cfg, base):
            return cfg
        assert isinstance(cfg, dict)
        return registry.build(cfg)

    return registry, config_type, build_func


BaseModule.build = classmethod(build)
