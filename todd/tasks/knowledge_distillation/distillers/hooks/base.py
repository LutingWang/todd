__all__ = [
    'BaseHook',
]

from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Literal

from torch import nn

from .....patches.py import get_
from .....utils import Status


class StatusEnum(IntEnum):
    INITIALIZED = auto()
    BOUND = auto()
    REGISTERED = auto()


class BaseHook(ABC):

    def __init__(
        self,
        *args,
        path: str,
        name: Literal['inputs', 'output'] | str = 'output',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._path = path
        self._name = name
        self._status = Status(StatusEnum.INITIALIZED)
        self.reset()

    def __call__(self):
        self._status.transit({StatusEnum.REGISTERED: StatusEnum.REGISTERED})
        return self._tensor()

    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> StatusEnum:
        return self._status.status

    @abstractmethod
    def _reset(self) -> None:
        pass

    @abstractmethod
    def _tensor(self):
        pass

    @abstractmethod
    def _register_tensor(self, tensor) -> None:
        pass

    def _forward_hook(self, module: nn.Module, inputs: tuple, output) -> None:
        if self._name == 'inputs':
            tensor = inputs
        elif self._name == 'output':
            tensor = output
        else:
            tensor = get_(module, self._name)
        self.register_tensor(tensor)

    def _bind(self, model: nn.Module) -> None:
        module: nn.Module = get_(model, self._path)
        self._handle = module.register_forward_hook(self._forward_hook)

    def _unbind(self) -> None:
        self._handle.remove()

    def reset(self) -> None:
        self._status.transit({
            StatusEnum.INITIALIZED: StatusEnum.INITIALIZED,
            StatusEnum.BOUND: StatusEnum.BOUND,
            StatusEnum.REGISTERED: StatusEnum.BOUND,
        })
        self._reset()

    def bind(self, model: nn.Module) -> None:
        self._status.transit({StatusEnum.INITIALIZED: StatusEnum.BOUND})
        self._bind(model)

    def unbind(self) -> None:
        self._status.transit({StatusEnum.BOUND: StatusEnum.INITIALIZED})
        self._unbind()

    def register_tensor(self, tensor) -> None:
        self._status.transit({
            StatusEnum.BOUND: StatusEnum.REGISTERED,
            StatusEnum.REGISTERED: StatusEnum.REGISTERED,
        })
        self._register_tensor(tensor)
