__all__ = [
    'BaseBuilder',
    'Builder',
    'NestedCollectionBuilder',
]

from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from ..configs import Config

if TYPE_CHECKING:
    from ...utils import NestedCollectionUtils

F = Callable[..., Any]


class BaseBuilder(ABC):

    def __init__(
        self,
        f: F,
        *args,
        priors: Iterable[str] | None = None,
        requires: Mapping[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.f = f
        self._priors = set() if priors is None else set(priors)
        self._requires = Config(requires)

    @property
    def priors(self) -> set[str]:
        return self._priors | self.requires

    @property
    def requires(self) -> set[str]:
        return set(self._requires.keys())

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
        return isinstance(obj, Config)

    def build(self, obj: Any, **kwargs) -> Any:
        return self.f(obj, **kwargs)


class NestedCollectionBuilder(BaseBuilder):

    def __init__(
        self,
        *args,
        utils: 'NestedCollectionUtils | None' = None,
        **kwargs,
    ) -> None:
        from ...utils import NestedCollectionUtils
        super().__init__(*args, **kwargs)
        if utils is None:
            utils = NestedCollectionUtils()
        utils.add_atomic_type(Config)
        self._utils = utils

    def should_build(self, obj: Any) -> bool:
        return self._utils.can_handle(obj)

    def build(self, obj: Any, **kwargs) -> Any:
        f = partial(self.f, **kwargs)
        return self._utils.map(f, obj)
