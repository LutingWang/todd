from abc import abstractmethod, abstractproperty
from typing import Any, Dict, Optional

import torch.nn as nn

from ..utils import getattr_recur


class BaseHook:
    def __init__(self, id_: str, path: str, on_input: bool = False, detach: bool = False):
        self._id = id_
        self._path = path
        self._on_input = on_input
        self.detach(detach)
        self.reset()

    @property
    def id_(self) -> str:
        return self._id

    @property
    def path(self) -> str:
        return self._path

    @property
    def on_input(self) -> bool:
        return self._on_input

    @abstractproperty
    def tensor(self) -> Dict[str, Any]:
        # TODO: raise error if returning None
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def register_tensor(self, _: Any):
        pass

    def detach(self, mode: bool = True):
        self._detach = mode

    def attach(self):
        self.detach(False)

    def _forward_hook(self, module: nn.Module, input_: Any, output: Any):
        self.register_tensor(input_ if self._on_input else output)

    def register_hook(self, model: nn.Module):
        module: nn.Module = getattr_recur(model, self.path)
        module.register_forward_hook(self._forward_hook)