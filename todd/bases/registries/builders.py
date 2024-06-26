__all__ = [
    'BaseBuilder',
    'Builder',
    'NestedCollectionBuilder',
]

from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from ...utils import NestedCollectionUtils

F = Callable[..., Any]


class BaseBuilder(ABC):

    def __init__(
        self,
        f: F,
        *args,
        requires: Mapping[str, str] | None = None,
        **kwargs,
    ) -> None:
        from ...configs import Config
        super().__init__(*args, **kwargs)
        self._f = f
        self._requires = Config(requires)

    @property
    def requires(self) -> tuple[str, ...]:
        return tuple(self._requires.keys())

    @abstractmethod
    def should_build(self, obj: Any) -> bool:
        pass

    @abstractmethod
    def build(self, obj: Any, **kwargs) -> Any:
        pass

    def __call__(self, obj: Any, **kwargs) -> Any:
        kwargs = {self._requires[k]: v for k, v in kwargs.items()}
        return self.build(obj, **kwargs)


class Builder(BaseBuilder):

    def should_build(self, obj: Any) -> bool:
        from ...configs import Config
        return isinstance(obj, Config)

    def build(self, obj: Any, **kwargs) -> Any:
        return self._f(obj, **kwargs)


class NestedCollectionBuilder(BaseBuilder):

    def __init__(
        self,
        *args,
        utils: 'NestedCollectionUtils | None' = None,
        **kwargs,
    ) -> None:
        from ...configs import Config
        from ...utils import NestedCollectionUtils
        super().__init__(*args, **kwargs)
        if utils is None:
            utils = NestedCollectionUtils()
        utils.add_atomic_type(Config)
        self._utils = utils

    def should_build(self, obj: Any) -> bool:
        return self._utils.can_handle(obj)

    def build(self, obj: Any, **kwargs) -> Any:
        f = partial(self._f, **kwargs)
        return self._utils.map(f, obj)
