__all__ = [
    'BaseHook',
]

from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Literal

from torch import nn

from todd.patches.py_ import get_
from todd.utils import StateMachine


class StateEnum(IntEnum):
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
        self._state = StateMachine(StateEnum.INITIALIZED)
        self.reset()

    def __call__(self):
        self._state.transit({StateEnum.REGISTERED: StateEnum.REGISTERED})
        return self._tensor()

    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> StateEnum:
        return self._state.state

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
        self._state.transit({
            StateEnum.INITIALIZED: StateEnum.INITIALIZED,
            StateEnum.BOUND: StateEnum.BOUND,
            StateEnum.REGISTERED: StateEnum.BOUND,
        })
        self._reset()

    def bind(self, model: nn.Module) -> None:
        self._state.transit({StateEnum.INITIALIZED: StateEnum.BOUND})
        self._bind(model)

    def unbind(self) -> None:
        self._state.transit({StateEnum.BOUND: StateEnum.INITIALIZED})
        self._unbind()

    def register_tensor(self, tensor) -> None:
        self._state.transit({
            StateEnum.BOUND: StateEnum.REGISTERED,
            StateEnum.REGISTERED: StateEnum.REGISTERED,
        })
        self._register_tensor(tensor)
