import functools
from abc import abstractmethod
from enum import IntEnum, auto
from typing import (
    Any,
    Callable,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    no_type_check,
)

import torch.nn as nn

from ..base import STEPS, Registry, getattr_recur

__all__ = [
    'BaseHook',
    'HOOKS',
]

T = TypeVar('T')


class _HookProto(Protocol):
    _path: str

    def register_tensor(self, tensor) -> None:
        pass


class Status(IntEnum):
    INITIALIZED = auto()
    BINDED = auto()
    REGISTERED = auto()


class _StatusMixin(_HookProto):

    def __init__(self) -> None:
        self._status = Status.INITIALIZED

    @property
    def status(self) -> Status:
        return self._status

    @staticmethod
    def transit(
        source: Union[Status, Sequence[Status]],
        target: Union[Status, Sequence[Status]],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if isinstance(source, Status):
            assert isinstance(target, Status)
            source = [source]
            target = [target]
        elif isinstance(target, Status):
            target = [target] * len(source)
        assert len(source) == len(target) == len(set(source)), (source, target)

        @no_type_check
        def wrapper(wrapped_func):

            @functools.wraps(wrapped_func)
            def wrapper_func(self: _StatusMixin, *args, **kwargs):
                if self._status not in source:
                    raise RuntimeError(
                        f"Hook {self._path} is not in {source}."
                    )
                self._status = target[source.index(self._status)]
                return wrapped_func(self, *args, **kwargs)

            return wrapper_func

        return wrapper


class _HookMixin(_HookProto):

    def _forward_hook(self, module: nn.Module, input_, output):
        self.register_tensor(output)

    def bind(self, model: nn.Module) -> None:
        module: nn.Module = getattr_recur(model, self._path)
        module.register_forward_hook(self._forward_hook)


class _TrackingMixin(_HookProto):

    def __init__(
        self,
        tracking_mode: bool = False,
    ) -> None:
        self._tracking_mode = tracking_mode

    @property
    def tracking_mode(self) -> bool:
        return self._tracking_mode

    def bind(self, model: nn.Module) -> None:
        self._model = model

    def track_tensor(self) -> None:
        if not self._tracking_mode:
            raise AttributeError(f"Hook {self._path} does not support track.")
        tensor = getattr_recur(self._model, self._path)
        self.register_tensor(tensor)


class BaseHook(_StatusMixin, _HookMixin, _TrackingMixin):

    def __init__(
        self,
        path: str,
        tracking_mode: bool = False,
    ) -> None:
        _StatusMixin.__init__(self)
        _HookMixin.__init__(self)
        _TrackingMixin.__init__(self, tracking_mode)
        self._path = path
        self._reset()

    def __call__(self):
        if self.status != Status.REGISTERED:
            raise RuntimeError(f"Hook {self._path} is not registered.")
        return self._tensor()

    @property
    def path(self) -> str:
        return self._path

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _register_tensor(self, tensor) -> None:
        pass

    @abstractmethod
    def _tensor(self):
        pass

    @_StatusMixin.transit(Status.REGISTERED, Status.BINDED)
    def reset(self) -> None:
        self._reset()

    @_StatusMixin.transit(Status.INITIALIZED, Status.BINDED)
    def bind(self, model: nn.Module) -> None:
        if self._tracking_mode:
            _TrackingMixin.bind(self, model)
        else:
            _HookMixin.bind(self, model)

    @_StatusMixin.transit(
        (Status.BINDED, Status.REGISTERED),
        Status.REGISTERED,
    )
    def register_tensor(self, tensor) -> None:
        self._register_tensor(tensor)


HOOKS: Registry[BaseHook] = Registry(
    'hooks',
    parent=STEPS,
    base=BaseHook,
)
