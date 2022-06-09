from typing import Callable, Tuple, Type, TypeVar, Union

from mmcv.runner import BaseModule
from mmcv.utils import Registry

T = TypeVar('T')


def build(cls, cfg, **kwargs) -> BaseModule:
    if cfg is None:
        return None
    module = cfg if isinstance(cfg, cls) else cls(cfg, **kwargs)
    return module


def build_metas(
    name: str,
    base: Type[T],
) -> Tuple[Registry, Callable[[Union[T, dict]], T]]:
    registry = Registry(name)

    def build_func(cfg: Union[T, dict]) -> T:
        if isinstance(cfg, base):
            return cfg
        assert isinstance(cfg, dict)
        return registry.build(cfg)

    return registry, build_func


BaseModule.build = classmethod(build)
